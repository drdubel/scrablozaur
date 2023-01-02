use regex::Regex;

fn main() {
    let text = "dead\ncos\ndeny\nzdeny\ndenyz\ndont\ndeck\n";
    let re = Regex::new(r"^de[a-z]{2}").unwrap();
    for mat in re.find_iter(text) {
        let substring = &text[mat.start()..mat.end()];
        println!("{}", substring);
    }
}
