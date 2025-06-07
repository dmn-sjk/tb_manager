use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers, MouseButton, MouseEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::{
    error::Error,
    fs::{self, remove_dir_all},
    io,
    os::unix::fs::symlink,
    path::Path,
    process::{Child, Command},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tempfile::TempDir;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
    Terminal,
};
use signal_hook::consts::SIGINT;
use signal_hook::flag;
use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the directory of folders with tb experiments
    #[arg(required = true)]
    log_dir: String,
}

// State for the UI
struct App {
    experiments: Vec<String>,
    selected: Vec<bool>,
    filter: String,
    filtered_experiments: Vec<usize>,
    list_state: ListState,
    focus: Focus,
    tensorboard: Option<Child>,
    log_dir: String,
    tmp_dir: TempDir,
    layout_chunks: Vec<Rect>,
}

enum Focus {
    List,
    Filter,
}

impl App {
    fn new(log_dir: String, experiments: Vec<String>, tmp_dir: TempDir) -> App {
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
            tensorboard: None,
            log_dir,
            tmp_dir,
            layout_chunks: Vec::new(),
        }
    }

    fn update_symlinks(&self) -> io::Result<()> {
        let tmp_path = self.tmp_dir.path();
        if tmp_path.exists() {
            remove_dir_all(tmp_path)?;
        }
        fs::create_dir(tmp_path)?;
        for (i, selected) in self.selected.iter().enumerate() {
            if *selected {
                let src = Path::new(&self.log_dir).join(&self.experiments[i]);
                let dst = tmp_path.join(&self.experiments[i]);
                symlink(&src, &dst)?;
            }
        }
        Ok(())
    }

    fn start_tensorboard(&mut self, port: u16) -> io::Result<()> {
        if let Some(mut child) = self.tensorboard.take() {
            child.kill()?;
        }
        let child = Command::new("tensorboard")
            .arg("--logdir")
            .arg(self.tmp_dir.path())
            .arg(format!("--port={}", port))
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;
        self.tensorboard = Some(child);
        Ok(())
    }

    fn update_filter(&mut self) {
        self.filtered_experiments = self
            .experiments
            .iter()
            .enumerate()
            .filter(|(_, exp)| exp.to_lowercase().contains(&self.filter.to_lowercase()))
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

fn get_experiment_folders(log_dir: &str) -> io::Result<Vec<String>> {
    let mut experiments = vec![];
    for entry in fs::read_dir(log_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                experiments.push(name.to_string());
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
    app.start_tensorboard(6006)?;

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
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),
                    Constraint::Min(0),
                    Constraint::Length(3),
                ])
                .split(f.size());
            app.layout_chunks = chunks.to_vec();

            // Status
            let status = Paragraph::new(format!(
                "Selected: {}",
                app.selected
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| **s)
                    .map(|(i, _)| &*app.experiments[i])
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .block(Block::default().borders(Borders::NONE));
            f.render_widget(status, chunks[0]);

            // Experiment list
            let items: Vec<ListItem> = app
                .filtered_experiments
                .iter()
                .map(|&i| {
                    let prefix = if app.selected[i] { "[x]" } else { "[ ]" };
                    ListItem::new(Line::from(vec![
                        Span::raw(prefix),
                        Span::raw(" "),
                        Span::raw(&app.experiments[i]),
                    ]))
                })
                .collect();
            let list = List::new(items)
                .block(Block::default().borders(Borders::ALL).title("Experiments"))
                .highlight_style(Style::default().fg(Color::Yellow));
            f.render_stateful_widget(list, chunks[1], &mut app.list_state);

            // Filter input
            let filter = Paragraph::new(app.filter.as_str())
                .block(Block::default().borders(Borders::ALL).title("Filter"));
            f.render_widget(filter, chunks[2]);

            // Set cursor for filter input
            if matches!(app.focus, Focus::Filter) {
                f.set_cursor(chunks[2].x + app.filter.len() as u16 + 1, chunks[2].y + 1);
            }
        })?;

        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) => {
                    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('q') {
                        break;
                    }
                    match app.focus {
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
                    }
                }
                Event::Mouse(mouse) => {
                    if mouse.kind == MouseEventKind::Up(MouseButton::Left) {
                        if app.layout_chunks.len() > 1 {
                            let list_chunk = app.layout_chunks[1];
                            // Check if click is in the experiment list area (chunks[1])
                            if mouse.row >= list_chunk.y && mouse.row < list_chunk.y + list_chunk.height {
                                let relative_row = (mouse.row - list_chunk.y - 1) as usize; // Adjust for border
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
                _ => {}
            }
        }
    }
    Ok(())
}
