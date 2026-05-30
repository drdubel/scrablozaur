fn main() {
    if let Err(err) = scrablozaur::main_cli() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
