use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

/// A fresh temp directory for one test's artifacts.
fn temp_dir(tag: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is before Unix epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("scrablozaur-cli-{tag}-{stamp}"));
    fs::create_dir_all(&dir).expect("create temporary test directory");
    dir
}

/// Run the CLI from the crate root, so its search for `languages/<code>.json`
/// finds the real definitions.
fn cli(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_scrablozaur"))
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .expect("run scrablozaur CLI")
}

fn assert_ok(output: &Output, what: &str) {
    assert!(
        output.status.success(),
        "{what} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn build(words: &Path, out: &Path, gaddag: bool) -> Output {
    let cmd = if gaddag { "build-gaddag" } else { "build" };
    cli(&[
        std::ffi::OsStr::new(cmd),
        words.as_os_str(),
        out.as_os_str(),
    ])
}

#[test]
fn build_command_creates_dawg_bin() {
    let dir = temp_dir("build");
    let words_path = dir.join("words.txt");
    let dawg_path = dir.join("dawg.bin");
    fs::write(&words_path, "ala\nal\nkot\n").expect("write test words");

    let output = build(&words_path, &dawg_path, false);
    assert_ok(&output, "build command");
    assert!(dawg_path.exists(), "dawg.bin was not created");

    let metadata = fs::metadata(&dawg_path).expect("read generated dawg.bin metadata");
    assert!(metadata.len() > 20, "generated dawg.bin looks too small");

    let header = fs::read(&dawg_path).expect("read generated dawg.bin");
    assert_eq!(&header[..8], b"SCRBDWG2", "missing format magic");
    assert_eq!(header[12], 0, "should be stamped as a DAWG");
}

/// A word list built into a DAWG must then be readable through `lookup`, which
/// is what exercises the header end to end -- the stamped alphabet has to be
/// accepted by the language the lookup runs under.
#[test]
fn a_built_dawg_round_trips_through_lookup() {
    let dir = temp_dir("roundtrip");
    let words_path = dir.join("words.txt");
    let dawg_path = dir.join("dawg.bin");
    fs::write(&words_path, "ala\nal\nkot\n").expect("write test words");
    assert_ok(&build(&words_path, &dawg_path, false), "build command");

    let found = cli(&[
        std::ffi::OsStr::new("lookup"),
        std::ffi::OsStr::new("pl"),
        dawg_path.as_os_str(),
        std::ffi::OsStr::new("kot"),
    ]);
    assert_ok(&found, "lookup of a present word");
    let stdout = String::from_utf8_lossy(&found.stdout);
    assert!(stdout.contains("found"), "unexpected output: {stdout}");
    assert!(!stdout.contains("not found"), "unexpected output: {stdout}");

    let missing = cli(&[
        std::ffi::OsStr::new("lookup"),
        std::ffi::OsStr::new("pl"),
        dawg_path.as_os_str(),
        std::ffi::OsStr::new("kotek"),
    ]);
    assert_ok(&missing, "lookup of an absent word");
    assert!(
        String::from_utf8_lossy(&missing.stdout).contains("not found"),
        "a word outside the lexicon should not be found"
    );
}

/// The header's whole purpose: a dictionary whose letters the language does not
/// have must be refused, not decoded into a silently wrong lexicon.
#[test]
fn a_dictionary_in_the_wrong_language_is_refused() {
    let dir = temp_dir("mismatch");
    let words_path = dir.join("words.txt");
    let dawg_path = dir.join("dawg.bin");
    // `quiz` needs q and z; Polish has no q.
    fs::write(&words_path, "quiz\ncat\n").expect("write test words");
    assert_ok(&build(&words_path, &dawg_path, false), "build command");

    let output = cli(&[
        std::ffi::OsStr::new("lookup"),
        std::ffi::OsStr::new("pl"),
        dawg_path.as_os_str(),
        std::ffi::OsStr::new("cat"),
    ]);
    assert!(
        !output.status.success(),
        "an english lexicon must not load as polish"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("different language") || stderr.contains('q'),
        "error should name the offending letter, got: {stderr}"
    );
}

/// `build-gaddag` stamps a different kind byte, and loading it where a DAWG is
/// wanted must say so rather than quietly finding no words.
#[test]
fn a_gaddag_is_refused_where_a_dawg_is_wanted() {
    let dir = temp_dir("kind");
    let words_path = dir.join("words.txt");
    let gaddag_path = dir.join("gaddag.bin");
    fs::write(&words_path, "ala\nal\nkot\n").expect("write test words");
    assert_ok(&build(&words_path, &gaddag_path, true), "build-gaddag");

    let header = fs::read(&gaddag_path).expect("read generated gaddag.bin");
    assert_eq!(header[12], 1, "should be stamped as a GADDAG");

    let output = cli(&[
        std::ffi::OsStr::new("lookup"),
        std::ffi::OsStr::new("pl"),
        gaddag_path.as_os_str(),
        std::ffi::OsStr::new("kot"),
    ]);
    assert!(
        !output.status.success(),
        "a GADDAG must not load where a DAWG is wanted"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("GADDAG"),
        "error should name the mismatch"
    );
}
