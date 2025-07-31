use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers, MouseButton, MouseEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::{
    error::Error,
    fs::{self, remove_dir_all},
    io::{self, BufRead, BufReader},
    os::unix::fs::symlink,
    path::Path,
    process::{Child, Command},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, SystemTime},
};
use tempfile::TempDir;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Scrollbar, ScrollbarState},
    Terminal,
};
use signal_hook::consts::SIGINT;
use signal_hook::flag;
use clap::Parser;
use chrono::{DateTime, Local};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the directory of folders with tb experiments
    #[arg(required = true)]
    log_dir: String,
}

struct Experiment {
    name: String,
    modified: SystemTime,
    modified_str: String,
}

#[derive(PartialEq, Clone, Copy)]
enum SortDirection {
    Asc,
    Desc,
}

enum SortMode {
    Alphabetical(SortDirection),
    Date(SortDirection),
}

#[derive(Clone, Copy)]
enum Tab {
    List,
    Tensorboard,
}

// State for the UI
struct App {
    experiments: Vec<Experiment>,
    selected: Vec<bool>,
    filter: String,
    filtered_experiments: Vec<usize>,
    list_state: ListState,
    focus: Focus,
    active_tab: Tab,
    tensorboard: Option<Child>,
    tensorboard_output: Vec<String>,
    tensorboard_output_rx: Option<mpsc::Receiver<String>>,
    tensorboard_scroll_state: ScrollbarState,
    tensorboard_scroll: u16,
    log_dir: String,
    tmp_dir: TempDir,
    layout_chunks: Vec<Rect>,
    header_chunks: Vec<Rect>,
    sort_mode: SortMode,
    tab_chunk: Rect,
}

enum Focus {
    List,
    Filter,
}

impl App {
    fn new(log_dir: String, experiments: Vec<Experiment>, tmp_dir: TempDir) -> App {
        let selected = vec![false; experiments.len()];
        let filtered_experiments = (0..experiments.len()).collect();
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        App {
            experiments,
            selected,
            filter: String::new(),
            filtered_experiments,
            list_state,
            focus: Focus::List,
            active_tab: Tab::List,
            tensorboard: None,
            tensorboard_output: Vec::new(),
            tensorboard_output_rx: None,
            tensorboard_scroll_state: ScrollbarState::default(),
            tensorboard_scroll: 0,
            log_dir,
            tmp_dir,
            layout_chunks: Vec::new(),
            header_chunks: Vec::new(),
            sort_mode: SortMode::Date(SortDirection::Desc),
            tab_chunk: Rect::default(),
        }
    }

    fn sort_experiments(&mut self) {
        match self.sort_mode {
            SortMode::Alphabetical(dir) => {
                self.experiments.sort_by(|a, b| {
                    if dir == SortDirection::Asc {
                        a.name.cmp(&b.name)
                    } else {
                        b.name.cmp(&a.name)
                    }
                });
            }
            SortMode::Date(dir) => {
                self.experiments.sort_by(|a, b| {
                    if dir == SortDirection::Asc {
                        a.modified.cmp(&b.modified)
                    } else {
                        b.modified.cmp(&a.modified)
                    }
                });
            }
        }
        self.update_filter();
    }

    fn set_sort_mode(&mut self, new_sort_mode: SortMode) {
        self.sort_mode = new_sort_mode;
        self.sort_experiments();
    }

    fn update_symlinks(&self) -> io::Result<()> {
        let tmp_path = self.tmp_dir.path();
        if tmp_path.exists() {
            remove_dir_all(tmp_path)?;
        }
        fs::create_dir(tmp_path)?;
        for (i, selected) in self.selected.iter().enumerate() {
            if *selected {
                let src = Path::new(&self.log_dir).join(&self.experiments[i].name);
                let dst = tmp_path.join(&self.experiments[i].name);
                symlink(&src, &dst)?;
            }
        }
        Ok(())
    }

    fn start_tensorboard(&mut self, port: u16, image_samples: u32) -> io::Result<()> {
        if let Some(mut child) = self.tensorboard.take() {
            child.kill()?;
        }

        let (tx, rx) = mpsc::channel();
        self.tensorboard_output_rx = Some(rx);
        self.tensorboard_output.clear();
        self.tensorboard_scroll = 0;
        self.tensorboard_scroll_state = ScrollbarState::default();

        let mut child = Command::new("tensorboard")
            .arg("--logdir")
            .arg(self.tmp_dir.path())
            .arg(format!("--samples_per_plugin=images={}", image_samples))
            .arg(format!("--port={}", port))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        let stdout = child.stdout.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "Failed to capture tensorboard stdout")
        })?;
        let tx_out = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if tx_out.send(line).is_err() {
                        break;
                    }
                }
            }
        });

        let stderr = child.stderr.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Other, "Failed to capture tensorboard stderr")
        })?;
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            }
        });

        self.tensorboard = Some(child);
        Ok(())
    }

    fn update_filter(&mut self) {
        self.filtered_experiments = self
            .experiments
            .iter()
            .enumerate()
            .filter(|(_, exp)| exp.name.to_lowercase().contains(&self.filter.to_lowercase()))
            .map(|(i, _)| i)
            .collect();
        if self.filtered_experiments.is_empty() {
            self.list_state.select(None);
        } else if self.list_state.selected().map_or(true, |i| i >= self.filtered_experiments.len()) {
            self.list_state.select(Some(0));
        }
    }

    fn list_next(&mut self) {
        if let Some(selected) = self.list_state.selected() {
            let next = (selected + 1) % self.filtered_experiments.len();
            self.list_state.select(Some(next));
        }
    }

    fn list_previous(&mut self) {
        if let Some(selected) = self.list_state.selected() {
            let next = if selected == 0 {
                self.filtered_experiments.len() - 1
            } else {
                selected - 1
            };
            self.list_state.select(Some(next));
        }
    }
}

fn check_tensorboard_exists() -> io::Result<()> {
    let output = Command::new("tensorboard").arg("--help").output();

    match output {
        Ok(output) => {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let err_msg = if stderr.contains("command not found") {
                    format!(
                        "TensorBoard command failed. If using pyenv, an active python environment with tensorboard might be missing.\n\nPyenv output:\n{}",
                        stderr
                    )
                } else {
                    format!("'tensorboard --help' failed with: {}", stderr)
                };
                Err(io::Error::new(io::ErrorKind::Other, err_msg))
            } else {
                Ok(())
            }
        }
        Err(e) => {
            if e.kind() == io::ErrorKind::NotFound {
                Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "TensorBoard not found. Please ensure it is installed and in your PATH.",
                ))
            } else {
                Err(e)
            }
        }
    }
}

fn get_experiment_folders(log_dir: &str) -> io::Result<Vec<Experiment>> {
    let mut experiments = vec![];
    for entry in fs::read_dir(log_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                let metadata = entry.metadata()?;
                let modified: SystemTime = metadata.modified()?;
                let datetime: DateTime<Local> = modified.into();
                experiments.push(Experiment {
                    name: name.to_string(),
                    modified,
                    modified_str: datetime.format("%Y-%m-%d %H:%M:%S").to_string(),
                });
            }
        }
    }
    if experiments.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "No experiment folders found"));
    }
    Ok(experiments)
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let log_dir_abs = fs::canonicalize(&cli.log_dir).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Error processing log directory '{}': {}", &cli.log_dir, e),
        )
    })?;
    let log_dir = log_dir_abs.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Log directory path contains invalid UTF-8",
        )
    })?;

    check_tensorboard_exists()?;

    // Get experiment folders
    let experiments = get_experiment_folders(log_dir)?;
    
    // Create temporary directory
    let tmp_dir = tempfile::tempdir_in("/tmp")?;
    
    // Setup signal handler for cleanup
    let term = Arc::new(AtomicBool::new(false));
    flag::register(SIGINT, Arc::clone(&term))?;

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run app
    let mut app = App::new(log_dir.to_string(), experiments, tmp_dir);
    app.sort_experiments();
    // TODO: add possibility to adjust those arguments in the app
    app.start_tensorboard(6006, 10000)?;

    let result = run_app(&mut terminal, &mut app, Arc::clone(&term));

    // Cleanup
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    if let Some(mut child) = app.tensorboard.take() {
        child.kill()?;
    }

    result
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    term: Arc<AtomicBool>,
) -> Result<(), Box<dyn Error>> {
    while !term.load(Ordering::Relaxed) {
        if let Some(rx) = &app.tensorboard_output_rx {
            for line in rx.try_iter() {
                app.tensorboard_output.push(line);
            }
            app.tensorboard_scroll_state = app.tensorboard_scroll_state.content_length(app.tensorboard_output.len());
        }

        terminal.draw(|f| {
            let footer_height = if matches!(app.active_tab, Tab::List) { 3 } else { 0 };
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1), // Tabs
                    Constraint::Min(0),    // Content
                    Constraint::Length(footer_height), // Optional footer
                ])
                .split(f.size());
            app.tab_chunk = chunks[0];

            let tab_titles = [
                Span::styled(
                    "List [1]",
                    if matches!(app.active_tab, Tab::List) {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default()
                    },
                ),
                Span::raw(" | "),
                Span::styled(
                    "Tensorboard [2]",
                    if matches!(app.active_tab, Tab::Tensorboard) {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default()
                    },
                ),
            ];
            let tabs = Paragraph::new(Line::from(tab_titles.to_vec()));
            f.render_widget(tabs, chunks[0]);

            match app.active_tab {
                Tab::List => {
                    app.header_chunks.clear();
                    let list_chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([
                            Constraint::Length(1), // Status
                            Constraint::Length(1), // Header
                            Constraint::Min(0),    // List
                        ])
                        .split(chunks[1]);

                    let status_chunk = list_chunks[0];
                    let header_chunk = list_chunks[1];
                    let list_chunk = list_chunks[2];
                    let filter_chunk = chunks[2];
                    app.layout_chunks = vec![status_chunk, header_chunk, list_chunk, filter_chunk];

                    // Status
                    let status = Paragraph::new(format!(
                        "Selected: {}",
                        app.selected
                            .iter()
                            .enumerate()
                            .filter(|(_, s)| **s)
                            .map(|(i, _)| &*app.experiments[i].name)
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                    .block(Block::default().borders(Borders::NONE));
                    f.render_widget(status, status_chunk);

                    // Header
                    let header_chunks = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Percentage(80), Constraint::Percentage(20)])
                        .split(header_chunk);
                    app.header_chunks = header_chunks.to_vec();

                    let name_sort_indicator = match app.sort_mode {
                        SortMode::Alphabetical(SortDirection::Asc) => "▲",
                        SortMode::Alphabetical(SortDirection::Desc) => "▼",
                        _ => " ",
                    };
                    let date_sort_indicator = match app.sort_mode {
                        SortMode::Date(SortDirection::Asc) => "▲",
                        SortMode::Date(SortDirection::Desc) => "▼",
                        _ => " ",
                    };

                    let name_header = Paragraph::new(format!("Name [n] {}", name_sort_indicator))
                        .block(Block::default().borders(Borders::NONE));
                    let date_header = Paragraph::new(format!("Date [d] {}", date_sort_indicator))
                        .block(Block::default().borders(Borders::NONE));
                    f.render_widget(name_header, header_chunks[0]);
                    f.render_widget(date_header, header_chunks[1]);

                    // Experiment list
                    let name_width = (list_chunk.width as f32 * 0.8).floor() as usize;
                    let date_width = (list_chunk.width as f32 * 0.2).floor() as usize;

                    let items: Vec<ListItem> = app
                        .filtered_experiments
                        .iter()
                        .map(|&i| {
                            let exp = &app.experiments[i];
                            let prefix = if app.selected[i] { "[x]" } else { "[ ]" };
                            let mut name = exp.name.clone();
                            if name.len() > name_width - 3 { // -3 for prefix and space
                                name.truncate(name_width - 6);
                                name.push_str("...");
                            }

                            ListItem::new(Line::from(vec![
                                Span::raw(prefix),
                                Span::raw(" "),
                                Span::raw(format!("{:<width$}", name, width = name_width - 3)),
                                Span::styled(
                                    format!("{:<width$}", &exp.modified_str, width = date_width),
                                    Style::default().fg(Color::DarkGray),
                                ),
                            ]))
                        })
                        .collect();
                    let list = List::new(items)
                        .block(Block::default().borders(Borders::NONE))
                        .highlight_style(Style::default().fg(Color::Yellow));
                    f.render_stateful_widget(list, list_chunk, &mut app.list_state);

                    // Filter input
                    let filter_chunk = chunks[2];
                    let filter = Paragraph::new(app.filter.as_str())
                        .block(Block::default().borders(Borders::ALL).title("Filter"));
                    f.render_widget(filter, filter_chunk);

                    // Set cursor for filter input
                    if matches!(app.focus, Focus::Filter) {
                        f.set_cursor(filter_chunk.x + app.filter.len() as u16 + 1, filter_chunk.y + 1);
                    }
                }
                Tab::Tensorboard => {
                    app.layout_chunks.clear();
                    app.header_chunks.clear();
                    let output_chunk = chunks[1];
                    let output_text: Vec<Line> =
                        app.tensorboard_output.iter().map(|s| Line::from(s.as_str())).collect();
                    let output_paragraph = Paragraph::new(output_text)
                        .block(Block::default().borders(Borders::ALL).title("Tensorboard Output"))
                        .scroll((app.tensorboard_scroll, 0));
                    f.render_widget(output_paragraph, output_chunk);
                    
                    app.tensorboard_scroll_state = app.tensorboard_scroll_state.viewport_content_length(output_chunk.height as usize);
                    f.render_stateful_widget(
                        Scrollbar::default()
                            .orientation(ratatui::widgets::ScrollbarOrientation::VerticalRight)
                            .begin_symbol(Some("↑"))
                            .end_symbol(Some("↓")),
                        output_chunk.inner(&ratatui::layout::Margin { vertical: 1, horizontal: 0 }),
                        &mut app.tensorboard_scroll_state,
                    );
                }
            }
        })?;

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => {
                    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('q') {
                        break;
                    }

                    match key.code {
                        KeyCode::Char('1') => {
                            app.active_tab = Tab::List;
                            continue;
                        }
                        KeyCode::Char('2') => {
                            app.active_tab = Tab::Tensorboard;
                            continue;
                        }
                        _ => {}
                    }

                    match app.active_tab {
                        Tab::List => match app.focus {
                            Focus::List => match key.code {
                                KeyCode::Enter => {
                                    if let Some(i) = app.list_state.selected() {
                                        if let Some(&exp_idx) = app.filtered_experiments.get(i) {
                                            app.selected[exp_idx] = !app.selected[exp_idx];
                                            app.update_symlinks()?;
                                            app.start_tensorboard(6006)?;
                                        }
                                    }
                                }
                                KeyCode::Down => app.list_next(),
                                KeyCode::Up => app.list_previous(),
                                KeyCode::Tab => {
                                    app.focus = Focus::Filter;
                                }
                                KeyCode::Char('n') => {
                                    let new_dir = if let SortMode::Alphabetical(dir) = app.sort_mode {
                                        if dir == SortDirection::Asc { SortDirection::Desc } else { SortDirection::Asc }
                                    } else {
                                        SortDirection::Asc
                                    };
                                    app.set_sort_mode(SortMode::Alphabetical(new_dir));
                                }
                                KeyCode::Char('d') => {
                                    let new_dir = if let SortMode::Date(dir) = app.sort_mode {
                                        if dir == SortDirection::Asc { SortDirection::Desc } else { SortDirection::Asc }
                                    } else {
                                        SortDirection::Desc
                                    };
                                    app.set_sort_mode(SortMode::Date(new_dir));
                                }
                                _ => {}
                            },
                            Focus::Filter => match key.code {
                                KeyCode::Char(c) => {
                                    app.filter.push(c);
                                    app.update_filter();
                                }
                                KeyCode::Backspace => {
                                    app.filter.pop();
                                    app.update_filter();
                                }
                                KeyCode::Tab => {
                                    app.focus = Focus::List;
                                }
                                _ => {}
                            },
                        },
                        Tab::Tensorboard => match key.code {
                            KeyCode::Down => {
                                app.tensorboard_scroll = app.tensorboard_scroll.saturating_add(1);
                                app.tensorboard_scroll_state = app.tensorboard_scroll_state.position(app.tensorboard_scroll as usize);
                            }
                            KeyCode::Up => {
                                app.tensorboard_scroll = app.tensorboard_scroll.saturating_sub(1);
                                app.tensorboard_scroll_state = app.tensorboard_scroll_state.position(app.tensorboard_scroll as usize);
                            }
                            _ => {}
                        },
                    }
                }
                Event::Mouse(mouse) => {
                    if mouse.kind == MouseEventKind::Up(MouseButton::Left) {
                        if mouse.row == app.tab_chunk.y {
                            let list_tab_width = "List [1]".len() as u16;
                            let separator_width = " | ".len() as u16;
                            let list_tab_end = app.tab_chunk.x + list_tab_width;
                            let tensorboard_tab_start = list_tab_end + separator_width;
                            let tensorboard_tab_width = "Tensorboard [2]".len() as u16;

                            if mouse.column >= app.tab_chunk.x && mouse.column < list_tab_end {
                                app.active_tab = Tab::List;
                            } else if mouse.column >= tensorboard_tab_start
                                && mouse.column < tensorboard_tab_start + tensorboard_tab_width
                            {
                                app.active_tab = Tab::Tensorboard;
                            }
                        }
                    }
                    match app.active_tab {
                        Tab::List => {
                            // Header clicks
                            if mouse.kind == MouseEventKind::Up(MouseButton::Left) && !app.header_chunks.is_empty() {
                                let name_header_chunk = app.header_chunks[0];
                                let date_header_chunk = app.header_chunks[1];

                                if mouse.row == name_header_chunk.y && mouse.column >= name_header_chunk.x && mouse.column < name_header_chunk.x + name_header_chunk.width {
                                    let new_dir = if let SortMode::Alphabetical(dir) = app.sort_mode {
                                        if dir == SortDirection::Asc { SortDirection::Desc } else { SortDirection::Asc }
                                    } else {
                                        SortDirection::Asc
                                    };
                                    app.set_sort_mode(SortMode::Alphabetical(new_dir));
                                } else if mouse.row == date_header_chunk.y && mouse.column >= date_header_chunk.x && mouse.column < date_header_chunk.x + date_header_chunk.width {
                                    let new_dir = if let SortMode::Date(dir) = app.sort_mode {
                                        if dir == SortDirection::Asc { SortDirection::Desc } else { SortDirection::Asc }
                                    } else {
                                        SortDirection::Desc
                                    };
                                    app.set_sort_mode(SortMode::Date(new_dir));
                                }
                            }

                            if mouse.kind == MouseEventKind::Up(MouseButton::Left) {
                                if app.layout_chunks.len() > 2 {
                                    let list_chunk = app.layout_chunks[2];
                                    // Check if click is in the experiment list area
                                    if mouse.row >= list_chunk.y && mouse.row < list_chunk.y + list_chunk.height {
                                        let relative_row = (mouse.row - list_chunk.y) as usize;
                                        if relative_row < app.filtered_experiments.len() {
                                            let exp_idx = app.filtered_experiments[relative_row];
                                            app.selected[exp_idx] = !app.selected[exp_idx];
                                            app.update_symlinks()?;
                                            app.start_tensorboard(6006)?;
                                            app.list_state.select(Some(relative_row)); // Update selection
                                        }
                                    }
                                }
                            }
                        }
                        Tab::Tensorboard => {
                            match mouse.kind {
                                MouseEventKind::ScrollDown => {
                                    app.tensorboard_scroll = app.tensorboard_scroll.saturating_add(1);
                                    app.tensorboard_scroll_state = app.tensorboard_scroll_state.position(app.tensorboard_scroll as usize);
                                }
                                MouseEventKind::ScrollUp => {
                                    app.tensorboard_scroll = app.tensorboard_scroll.saturating_sub(1);
                                    app.tensorboard_scroll_state = app.tensorboard_scroll_state.position(app.tensorboard_scroll as usize);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}
