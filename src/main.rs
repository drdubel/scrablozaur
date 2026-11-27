fn main() {
    if let Err(err) = scrablozaur_core::main_cli() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
