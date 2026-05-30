use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn build_command_creates_dawg_bin() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is before Unix epoch")
        .as_nanos();
    let temp_root = std::env::temp_dir().join(format!("scrablozaur-cli-test-{stamp}"));
    fs::create_dir_all(&temp_root).expect("create temporary test directory");

    let words_path = temp_root.join("words.txt");
    let dawg_path = temp_root.join("dawg.bin");
    fs::write(&words_path, "ala\nal\nkot\n").expect("write test words");

    let output = Command::new(env!("CARGO_BIN_EXE_scrablozaur"))
        .arg("build")
        .arg(&words_path)
        .arg(&dawg_path)
        .output()
        .expect("run scrablozaur CLI");

    assert!(
        output.status.success(),
        "build command failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(dawg_path.exists(), "dawg.bin was not created");

    let metadata = fs::metadata(&dawg_path).expect("read generated dawg.bin metadata");
    assert!(metadata.len() > 8, "generated dawg.bin looks too small");
}
