//! The game layer against its specification: the web app's Python.
//!
//! `tests/fixtures/web_parity.json` is recorded from the real web app by
//! `tests/web_parity.py` -- whole games through its HTTP API with the
//! engine's RNG swapped in, bot decisions and hints on fixed positions, and the
//! scan dictionary pass on corrupted boards. Everything here replays it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use serde::Deserialize;

use super::*;
use crate::scan::{flag_invalid, verify_and_correct};
use crate::{leave_value, rack_after, Board, Language, LeaveNet, Lexicon, BOARD_SIZE};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The real dictionaries and nets, loaded once per test binary.
fn pack(code: &str) -> LanguagePack {
    static PACKS: OnceLock<Mutex<HashMap<String, LanguagePack>>> = OnceLock::new();
    let packs = PACKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut packs = packs.lock().unwrap();
    packs
        .entry(code.to_string())
        .or_insert_with(|| {
            let r = root();
            let text = std::fs::read_to_string(r.join(format!("languages/{code}.json"))).unwrap();
            let lang = Language::from_json(&text).unwrap();
            let words = r.join(format!("words/{code}"));
            let lexicon = Lexicon::open(
                lang,
                words.join("dawg.bin").to_str().unwrap(),
                Some(words.join("gaddag.bin").to_str().unwrap()),
            )
            .unwrap();
            let net_path = r.join(format!("smart_player/models/{code}/leave_value.bin"));
            let net = LeaveNet::open(lang, net_path.to_str().unwrap()).unwrap();
            LanguagePack::new(Arc::new(lexicon), Some(Arc::new(net))).unwrap()
        })
        .clone()
}

// ── Fixture schema (see tests/web_parity.py) ─────────────────────────────────

type HintRow = (String, u32, usize, usize, bool, f64);

#[derive(Deserialize)]
struct Fixture {
    version: u32,
    rank_windows: Vec<(usize, usize)>,
    strength_windows: Vec<(usize, usize)>,
    decisions: Vec<Decision>,
    scan_cases: Vec<ScanCase>,
    traces: Vec<Trace>,
}

#[derive(Deserialize)]
struct Decision {
    lang: String,
    rows: Vec<String>,
    blanks: Vec<(usize, usize)>,
    rack: String,
    bag_remaining: usize,
    choose_move: Choice,
    score_hints: Vec<HintRow>,
    smart_hints: Vec<HintRow>,
    ratings: Vec<(u32, u8)>,
}

#[derive(Deserialize)]
struct Choice {
    word: String,
    score: u32,
    row: usize,
    col: usize,
    horizontal: bool,
    endgame_diff: Option<i32>,
    endgame_exact: Option<bool>,
}

#[derive(Deserialize)]
struct ScanCase {
    lang: String,
    rows: Vec<String>,
    alternatives: Vec<(usize, usize, String)>,
    locked: Vec<(usize, usize)>,
    flag_only: Vec<(usize, usize)>,
    corrected: Vec<String>,
    flagged: Vec<(usize, usize)>,
}

#[derive(Deserialize)]
struct Trace {
    name: String,
    lang: String,
    seed: u64,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    action: String,
    args: serde_json::Value,
    error: Option<String>,
    suggestions: Option<Vec<HintRow>>,
    state: Option<StateRow>,
    preview: Option<serde_json::Value>,
}

type MoveRow = (String, u32, usize, usize, bool, bool);
type PlayRow = (String, u32, usize, usize, bool, String);
type EquityRow = (f64, f64, Option<PlayRow>, Option<String>);
type EndgameRow = (i32, i32, Option<PlayRow>, bool);

/// A logged turn as the fixture writes it: player, action, word, score, row,
/// col, horizontal, tiles, rack, rating, best, equity, endgame.
#[derive(Deserialize, Debug, PartialEq, Clone)]
struct TurnRow(
    usize,
    String,
    String,
    u32,
    usize,
    usize,
    bool,
    usize,
    String,
    Option<u8>,
    Option<PlayRow>,
    Option<EquityRow>,
    Option<EndgameRow>,
);

#[derive(Deserialize, Debug, PartialEq)]
struct StateRow {
    board: Vec<String>,
    blanks: Vec<(usize, usize)>,
    owners: Vec<String>,
    players: Vec<(String, bool, i32, String)>,
    current: usize,
    first: bool,
    move_number: u32,
    can_undo: bool,
    mode: String,
    tiles_remaining: usize,
    last_computer_move: Option<MoveRow>,
    game_over: bool,
    winner: Option<String>,
    rating: Option<u8>,
    turns: Vec<TurnRow>,
    unseen: Vec<(String, u8)>,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let text = std::fs::read_to_string(root().join("tests/fixtures/web_parity.json"))
            .expect("tests/fixtures/web_parity.json -- run `uv run python tests/web_parity.py`");
        let f: Fixture = serde_json::from_str(&text).unwrap();
        assert_eq!(f.version, 2, "fixture format");
        f
    })
}

fn grid_of(rows: &[String]) -> [[char; BOARD_SIZE]; BOARD_SIZE] {
    let mut g = [['-'; BOARD_SIZE]; BOARD_SIZE];
    for (r, row) in rows.iter().enumerate() {
        for (c, ch) in row.chars().enumerate() {
            g[r][c] = ch;
        }
    }
    g
}

fn board_of(lang: &'static Language, rows: &[String], blanks: &[(usize, usize)]) -> Board {
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|row| row.chars().map(String::from).collect())
        .collect();
    let mut mask = vec![vec![false; BOARD_SIZE]; BOARD_SIZE];
    for &(r, c) in blanks {
        mask[r][c] = true;
    }
    Board::from_grid(lang, &cells, Some(&mask)).unwrap()
}

fn hint_row(s: &Suggestion) -> HintRow {
    (s.word.clone(), s.score, s.row, s.col, s.horizontal, s.value)
}

// ── Decisions ────────────────────────────────────────────────────────────────

#[test]
fn rank_windows_match_the_web() {
    let f = fixture();
    for (level, &want) in (1..=10).zip(&f.rank_windows) {
        assert_eq!(rank_window(level), want, "level {level}");
    }
    for (strength, &want) in (1..=100).zip(&f.strength_windows) {
        assert_eq!(strength_rank_window(strength), want, "strength {strength}");
    }
}

/// `choose_move`'s value of a play in this position, for telling a genuine
/// disagreement from a float tie between PyTorch and the engine's net.
fn static_value(board: &Board, net: &LeaveNet, rack: &str, score: u32, used: &[char]) -> f64 {
    let feats = board.board_features();
    let unseen: usize = board
        .unseen_tile_counts(rack)
        .iter()
        .map(|&c| c as usize)
        .sum();
    let leave = rack_after(rack, used);
    score as f64 + leave_value(net, board.lang, &leave, unseen, &feats) as f64
}

#[test]
fn bot_decisions_match_the_web() {
    let mut checked = 0;
    let mut near_ties = 0;
    for d in &fixture().decisions {
        let pack = pack(&d.lang);
        let board = board_of(pack.lang, &d.rows, &d.blanks);
        let net = pack.leave_net.as_deref().unwrap();
        let opts = StaticOptions {
            endgame_max_nodes: INTERACTIVE_ENDGAME_NODES,
            ..StaticOptions::default()
        };
        let got = choose_move(&board, &pack.lexicon, net, &d.rack, d.bag_remaining, &opts).unwrap();
        let want = &d.choose_move;
        assert_eq!(got.endgame_diff, want.endgame_diff, "rack {}", d.rack);
        assert_eq!(got.endgame_exact, want.endgame_exact, "rack {}", d.rack);
        let (word, score, pos, used) =
            got.play
                .clone()
                .unwrap_or((String::new(), 0, (0, 0, true), Vec::new()));
        if (word.as_str(), score, pos)
            != (
                want.word.as_str(),
                want.score,
                (want.row, want.col, want.horizontal),
            )
        {
            // Only acceptable as a tie the two nets break differently.
            assert!(
                want.endgame_diff.is_none(),
                "endgame choice differs for rack {}",
                d.rack
            );
            let want_used =
                board.hand_tiles_for_word(&want.word, want.row, want.col, want.horizontal, &d.rack);
            let a = static_value(&board, net, &d.rack, score, &used);
            let b = static_value(&board, net, &d.rack, want.score, &want_used);
            assert!(
                (a - b).abs() < 1e-3,
                "rack {}: chose {word} ({a}) where the web chose {} ({b})",
                d.rack,
                want.word
            );
            near_ties += 1;
        }
        checked += 1;
    }
    eprintln!("bot decisions: {checked} positions, {near_ties} near-ties");
    assert!(checked >= 50, "only {checked} positions");
    assert!(
        near_ties <= 2,
        "{near_ties} near-ties: the nets disagree more than float noise"
    );
}

/// Smart values are rounded to a tenth before sorting, so a value within float
/// noise of a .05 boundary can round the other way; anything else must match.
fn assert_hints_match(got: &[HintRow], want: &[HintRow], what: &str) {
    if got == want {
        return;
    }
    eprintln!("{what}: equal within rounding, not exactly");
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    // Both lists are sorted by value, so position by position the values must
    // agree to within that rounding -- which still allows two near-equal plays
    // to swap places, and nothing else.
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g.5 - w.5).abs() <= 0.1 + 1e-9,
            "{what}: order differs at {g:?} vs {w:?}"
        );
    }
    let key = |h: &HintRow| (h.0.clone(), h.1, h.2, h.3, h.4);
    let mut by_key: HashMap<_, f64> = want.iter().map(|h| (key(h), h.5)).collect();
    for h in got {
        let w = by_key
            .remove(&key(h))
            .unwrap_or_else(|| panic!("{what}: unexpected {h:?} (wanted {want:?})"));
        assert!(
            (w - h.5).abs() <= 0.1 + 1e-9,
            "{what}: {h:?} valued {w} by the web"
        );
    }
}

#[test]
fn hints_and_ratings_match_the_web() {
    for d in &fixture().decisions {
        let pack = pack(&d.lang);
        let board = board_of(pack.lang, &d.rows, &d.blanks);
        let net = pack.leave_net.as_deref();
        let score: Vec<HintRow> = rank_suggestions(
            &board,
            &pack.lexicon,
            net,
            &d.rack,
            10,
            SortMode::Score,
            100,
            0,
        )
        .unwrap()
        .iter()
        .map(hint_row)
        .collect();
        assert_eq!(score, d.score_hints, "score hints for {}", d.rack);
        let smart: Vec<HintRow> = rank_suggestions(
            &board,
            &pack.lexicon,
            net,
            &d.rack,
            10,
            SortMode::Smart,
            100,
            0,
        )
        .unwrap()
        .iter()
        .map(hint_row)
        .collect();
        assert_hints_match(
            &smart,
            &d.smart_hints,
            &format!("smart hints for {}", d.rack),
        );
        for &(actual, want) in &d.ratings {
            assert_eq!(
                move_rating(&board, &pack.lexicon, &d.rack, actual),
                want,
                "rating {actual} for {}",
                d.rack
            );
        }
    }
}

// ── The scan dictionary pass ─────────────────────────────────────────────────

#[test]
fn scan_correction_matches_the_web() {
    for case in &fixture().scan_cases {
        let pack = pack(&case.lang);
        let mut grid = grid_of(&case.rows);
        let mut locked = [[false; BOARD_SIZE]; BOARD_SIZE];
        for &(r, c) in &case.locked {
            locked[r][c] = true;
        }
        let mut alternatives: crate::scan::Grid<Vec<char>> =
            std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
        for (r, c, alts) in &case.alternatives {
            alternatives[*r][*c] = alts.chars().collect();
        }
        let cells = |flags: [[bool; BOARD_SIZE]; BOARD_SIZE]| -> Vec<(usize, usize)> {
            let mut v = Vec::new();
            for (r, row) in flags.iter().enumerate() {
                for (c, &flagged) in row.iter().enumerate() {
                    if flagged {
                        v.push((r, c));
                    }
                }
            }
            v
        };
        assert_eq!(
            cells(flag_invalid(&pack.lexicon, &grid, &locked)),
            case.flag_only
        );
        let flagged = verify_and_correct(&pack.lexicon, &mut grid, &alternatives, &locked);
        assert_eq!(grid, grid_of(&case.corrected), "corrected grid");
        assert_eq!(cells(flagged), case.flagged);
    }
}

// ── Whole games ──────────────────────────────────────────────────────────────

fn error_kind(e: &GameError) -> &'static str {
    match e {
        GameError::NotInDictionary(_) => "NotInDictionary",
        GameError::FirstMoveMustCoverCentre => "FirstMoveMustCoverCentre",
        GameError::NotConnected => "NotConnected",
        GameError::CrossWordInvalid(_) => "CrossWordInvalid",
        GameError::MissingTiles => "MissingTiles",
        GameError::GameOver => "GameOver",
        GameError::NothingToUndo => "NothingToUndo",
        GameError::WrongMode => "WrongMode",
        other => panic!("the web has no equivalent of {other:?}"),
    }
}

fn state_row(session: &GameSession) -> StateRow {
    let s = session.state();
    let mut blanks = Vec::new();
    for r in 0..BOARD_SIZE {
        for c in 0..BOARD_SIZE {
            if s.blanks[r][c] {
                blanks.push((r, c));
            }
        }
    }
    StateRow {
        board: s.board.iter().map(|row| row.iter().collect()).collect(),
        blanks,
        owners: s
            .tile_owners
            .iter()
            .map(|row| {
                row.iter()
                    .map(|o| o.map_or('.', |i| char::from(b'0' + i)))
                    .collect()
            })
            .collect(),
        players: s
            .players
            .iter()
            .map(|p| (p.name.clone(), p.is_computer, p.score, p.letters.clone()))
            .collect(),
        current: s.current_player,
        first: s.is_first_move,
        move_number: s.move_number,
        can_undo: s.can_undo,
        mode: match s.mode {
            GameMode::Sandbox => "sandbox",
            GameMode::SandboxAuto => "sandbox_auto",
            GameMode::Competitive => "competitive",
            GameMode::Multiplayer => "multiplayer",
        }
        .to_string(),
        tiles_remaining: s.tiles_remaining,
        last_computer_move: s
            .last_computer_move
            .map(|m| (m.word, m.score, m.row, m.col, m.horizontal, m.passed)),
        game_over: s.game_over,
        winner: s.winner.map(|w| match w {
            Winner::Tie => "Remis".to_string(),
            Winner::Player(i) => s.players[i].name.clone(),
        }),
        rating: s.last_move_rating,
        turns: s.turns.iter().map(turn_row).collect(),
        unseen: s
            .unseen
            .iter()
            .map(|&(ch, n)| (ch.to_string(), n))
            .collect(),
    }
}

fn play_row(p: &PlayRecord) -> PlayRow {
    (
        p.word.clone(),
        p.score,
        p.row,
        p.col,
        p.horizontal,
        p.used.clone(),
    )
}

fn turn_row(t: &TurnRecord) -> TurnRow {
    let action = match t.action {
        TurnAction::Play => "play",
        TurnAction::Exchange => "exchange",
        TurnAction::Pass => "pass",
        TurnAction::Resign => "resign",
    };
    TurnRow(
        t.player,
        action.to_string(),
        t.word.clone(),
        t.score,
        t.row,
        t.col,
        t.horizontal,
        t.tiles,
        t.rack.clone(),
        t.rating,
        t.best.as_ref().map(play_row),
        t.equity.as_ref().map(|e| {
            let best = e.best_play.as_ref().map(play_row);
            (e.played, e.best, best, e.best_exchange.clone())
        }),
        t.endgame.as_ref().map(|e| {
            let best = e.best_play.as_ref().map(play_row);
            (e.played, e.best, best, e.exact)
        }),
    )
}

/// The logged turns agree exactly but for equity, which the web values with
/// PyTorch and the port with its own copy of the net: to a tenth, as smart
/// hints do. A different best turn passes only as a tie within that noise --
/// counted in `near_ties`, which the caller bounds.
fn assert_turns_match(got: &[TurnRow], want: &[TurnRow], at: &str, near_ties: &mut usize) {
    assert_eq!(got.len(), want.len(), "{at}: turns");
    for (g, w) in got.iter().zip(want) {
        let without_equity = |t: &TurnRow| TurnRow {
            11: None,
            ..t.clone()
        };
        assert_eq!(without_equity(g), without_equity(w), "{at}");
        match (&g.11, &w.11) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                let close = |x: f64, y: f64| (x - y).abs() <= 0.1 + 1e-9;
                assert!(
                    close(a.0, b.0) && close(a.1, b.1),
                    "{at}: equity {a:?} vs {b:?}"
                );
                if (&a.2, &a.3) != (&b.2, &b.3) {
                    *near_ties += 1;
                }
            }
            (a, b) => panic!("{at}: equity {a:?} vs {b:?}"),
        }
    }
}

/// What `_players_from_request` builds from a new-game request.
fn players_from_request(args: &serde_json::Value) -> (GameMode, Vec<PlayerConfig>) {
    let mode = match args["game_mode"].as_str().unwrap() {
        "competitive" => GameMode::Competitive,
        "sandbox_auto" => GameMode::SandboxAuto,
        _ => GameMode::Sandbox,
    };
    let level = |v: &serde_json::Value| v.as_u64().unwrap_or(DEFAULT_LEVEL as u64) as u8;
    let strength = |v: &serde_json::Value| v.as_u64().unwrap_or(100) as u8;
    let requested = args["players"].as_array().unwrap();
    let players = match mode {
        GameMode::Competitive => vec![
            PlayerConfig::human(requested[0]["name"].as_str().unwrap()),
            PlayerConfig::computer(
                "Komputer",
                level(&args["difficulty"]),
                strength(&args["strength"]),
            ),
        ],
        _ => requested
            .iter()
            .map(|p| PlayerConfig {
                name: p["name"].as_str().unwrap().to_string(),
                is_computer: p["is_computer"].as_bool().unwrap_or(false),
                difficulty: level(&p["difficulty"]),
                strength: strength(&p["strength"]),
            })
            .collect(),
    };
    (mode, players)
}

fn placement(args: &serde_json::Value) -> (String, usize, usize, bool) {
    (
        args["word"].as_str().unwrap().to_string(),
        args["row"].as_u64().unwrap() as usize,
        args["col"].as_u64().unwrap() as usize,
        args["horizontal"].as_bool().unwrap(),
    )
}

fn sort_of(args: &serde_json::Value) -> SortMode {
    match args["sort"].as_str().unwrap_or("score") {
        "smart" => SortMode::Smart,
        "sim" => SortMode::Sim,
        _ => SortMode::Score,
    }
}

/// The web plays a competitive computer's reply inside the human's request.
fn reply_if_due(session: &mut GameSession) {
    if session.computer_to_move() {
        session.computer_turn().unwrap();
    }
}

#[test]
fn whole_games_match_the_web() {
    let mut near_ties = 0;
    for trace in &fixture().traces {
        let pack = pack(&trace.lang);
        let mut session: Option<GameSession> = None;
        for (i, step) in trace.steps.iter().enumerate() {
            let at = format!("{} step {i} ({})", trace.name, step.action);
            let args = &step.args;
            let result: Result<(), GameError> = match step.action.as_str() {
                "new" => {
                    let (mode, players) = players_from_request(args);
                    let mut s = GameSession::new(pack.clone(), mode, players, trace.seed).unwrap();
                    reply_if_due(&mut s);
                    session = Some(s);
                    Ok(())
                }
                action => {
                    let s = session.as_mut().unwrap();
                    match action {
                        "human_move" => {
                            let (w, r, c, h) = placement(args);
                            s.human_move(&w, r, c, h).map(|_| reply_if_due(s))
                        }
                        "skip" => s.skip().map(|_| reply_if_due(s)),
                        "undo" => s.undo(),
                        "resign" => s.resign(),
                        "next_move" => s.next_auto_move().map(|_| ()),
                        "set_letters" => s.set_letters(args["letters"].as_str().unwrap()),
                        "computer_move" => {
                            let (w, r, c, h) = placement(args);
                            s.place_computer_word(&w, r, c, h).map(|_| ())
                        }
                        "hints" | "suggest" => {
                            let got = if action == "hints" {
                                s.hints(sort_of(args))
                            } else {
                                s.sandbox_suggestions(sort_of(args))
                            };
                            got.map(|list| {
                                let rows: Vec<HintRow> = list.iter().map(hint_row).collect();
                                assert_hints_match(&rows, step.suggestions.as_ref().unwrap(), &at);
                            })
                        }
                        "preview" => {
                            let (w, r, c, h) = placement(args);
                            let want = step.preview.as_ref().unwrap();
                            let got = match s.preview_score(&w, r, c, h) {
                                Preview::Score(n) => serde_json::json!({"score": n, "error": null}),
                                Preview::NotInDictionary => {
                                    serde_json::json!({"score": null, "error": "not_in_dict"})
                                }
                                Preview::InvalidPlacement => {
                                    serde_json::json!({"score": null, "error": "invalid_placement"})
                                }
                                Preview::ScoreError => {
                                    serde_json::json!({"score": null, "error": "score_error"})
                                }
                            };
                            assert_eq!(&got, want, "{at}");
                            Ok(())
                        }
                        other => panic!("unknown action {other}"),
                    }
                }
            };
            match (&result, &step.error) {
                (Ok(()), None) => {}
                (Err(e), Some(want)) => assert_eq!(error_kind(e), want, "{at}"),
                (Ok(()), Some(want)) => {
                    panic!("{at}: the web refused with {want}, the port accepted")
                }
                (Err(e), None) => panic!("{at}: the port refused with {e:?}, the web accepted"),
            }
            if let Some(want) = &step.state {
                let mut got = state_row(session.as_ref().unwrap());
                assert_turns_match(&got.turns, &want.turns, &at, &mut near_ties);
                got.turns = want.turns.clone();
                assert_eq!(&got, want, "{at}");
            }
        }
    }
    // Each review is checked once a state, so one tie shows up in every
    // state after it: bounded loosely, but not hidden.
    eprintln!("turn log: {near_ties} near-tie best turns");
    assert!(
        near_ties <= 60,
        "{near_ties} near ties: more than float noise"
    );
}

// ── Properties the web never had a test for ──────────────────────────────────

fn tiles_in_play(session: &GameSession) -> usize {
    let state = session.state();
    let on_board = state.board.iter().flatten().filter(|&&c| c != '-').count();
    let in_racks: usize = session
        .players()
        .iter()
        .map(|p| p.letters.chars().count())
        .sum();
    on_board + in_racks + session.bag_remaining()
}

#[test]
fn a_bot_game_conserves_tiles_and_ends() {
    let pack = pack("pl");
    let players = vec![
        PlayerConfig::computer("A", 9, 100),
        PlayerConfig::computer("B", 6, 70),
    ];
    let mut session = GameSession::new(pack.clone(), GameMode::SandboxAuto, players, 42).unwrap();
    let total = pack.lang.bag().len();
    for _ in 0..200 {
        if session.is_over() {
            break;
        }
        session.next_auto_move().unwrap();
        assert_eq!(
            tiles_in_play(&session),
            total,
            "a tile appeared or vanished"
        );
    }
    assert!(
        session.is_over(),
        "the game should end well inside 200 moves"
    );
}

#[test]
fn a_saved_game_restores_exactly() {
    let pl = pack("pl");
    let players = vec![
        PlayerConfig::human("Ala"),
        PlayerConfig::computer("Bot", 8, 100),
    ];
    let mut session = GameSession::new(pl.clone(), GameMode::Competitive, players, 7).unwrap();
    reply_if_due(&mut session);
    for _ in 0..6 {
        let hint = session.hints(SortMode::Score).unwrap();
        let Some(top) = hint.first() else { break };
        session
            .human_move(&top.word, top.row, top.col, top.horizontal)
            .unwrap();
        reply_if_due(&mut session);
    }
    let saved = session.to_json();
    let mut restored = GameSession::from_json(pl.clone(), &saved).unwrap();
    assert_eq!(restored.state(), session.state());
    assert_eq!(restored.to_json(), saved, "a round trip must be lossless");
    // Undo history and RNG come back too: both sessions continue identically.
    session.undo().unwrap();
    restored.undo().unwrap();
    assert_eq!(restored.state(), session.state());
    // A save in another language is refused, not misread.
    assert!(matches!(
        GameSession::from_json(pack("en"), &saved),
        Err(GameError::InvalidSave(_))
    ));
}

/// Each seat's logged scores add up to its score: the log misses no turn.
fn assert_log_adds_up(session: &GameSession) {
    let state = session.state();
    for (i, p) in state.players.iter().enumerate() {
        let logged: u32 = state
            .turns
            .iter()
            .filter(|t| t.player == i)
            .map(|t| t.score)
            .sum();
        assert_eq!(logged as i32, p.score, "{}'s turns", p.name);
    }
}

#[test]
fn the_turn_log_follows_the_game_through_undo_and_saves() {
    let pl = pack("pl");
    let players = vec![
        PlayerConfig::human("Ala"),
        PlayerConfig::computer("Bot", 8, 100),
    ];
    let mut session = GameSession::new(pl.clone(), GameMode::Competitive, players, 7).unwrap();
    reply_if_due(&mut session);
    let me = 0;

    // The best play rates 100 and is its own best...
    let rack = session.players()[me].letters.clone();
    let hints = session.hints(SortMode::Score).unwrap();
    let top = hints[0].clone();
    session
        .human_move(&top.word, top.row, top.col, top.horizontal)
        .unwrap();
    let turn = session
        .state()
        .turns
        .into_iter()
        .rfind(|t| t.player == me)
        .unwrap();
    assert_eq!(turn.action, TurnAction::Play);
    assert_eq!(
        (turn.word.as_str(), turn.score),
        (top.word.as_str(), top.score)
    );
    assert_eq!(
        (turn.rack.as_str(), turn.tiles),
        (rack.as_str(), top.used.len())
    );
    assert_eq!(turn.rating, Some(100));
    let best = turn.best.unwrap();
    assert_eq!(best.used, top.used.iter().collect::<String>());
    assert_eq!(
        (best.word, best.score, best.row, best.col, best.horizontal),
        (top.word, top.score, top.row, top.col, top.horizontal)
    );
    reply_if_due(&mut session);

    // ...a weaker one is measured against the best there was.
    let hints = session.hints(SortMode::Score).unwrap();
    let (top, weak) = (&hints[0], hints.last().unwrap());
    session
        .human_move(&weak.word, weak.row, weak.col, weak.horizontal)
        .unwrap();
    let turn = session.state().turns.last().unwrap().clone();
    assert_eq!(turn.rating, session.state().last_move_rating);
    assert_eq!(turn.best.unwrap().score, top.score);
    reply_if_due(&mut session);
    assert_log_adds_up(&session);

    // While the game is on, the computer's racks stay hidden.
    let state = session.state();
    assert!(state.turns.iter().any(|t| t.player != me));
    for t in &state.turns {
        assert_eq!(t.rack.is_empty(), t.player != me, "turn {t:?}");
    }

    // An exchange and a skip are turns too; undo takes the skip back, and the
    // computer's reply with it.
    let two: String = session.players()[me].letters.chars().take(2).collect();
    session.exchange(&two).unwrap();
    let exchange = session.state().turns.last().unwrap().clone();
    assert_eq!(
        (exchange.action, exchange.tiles, exchange.score),
        (TurnAction::Exchange, 2, 0)
    );
    reply_if_due(&mut session);
    let before_skip = session.state().turns.len();
    session.skip().unwrap();
    assert_eq!(session.state().turns[before_skip].action, TurnAction::Pass);
    reply_if_due(&mut session);
    session.undo().unwrap();
    assert_eq!(session.state().turns.len(), before_skip);
    assert_log_adds_up(&session);

    // A save carries the log.
    let restored = GameSession::from_json(pl.clone(), &session.to_json()).unwrap();
    assert_eq!(restored.state().turns, session.state().turns);

    // Resigning ends the game, and then every rack may be seen.
    session.resign().unwrap();
    let state = session.state();
    assert!(state.game_over);
    assert_eq!(state.turns.last().unwrap().action, TurnAction::Resign);
    assert!(state.turns.iter().all(|t| !t.rack.is_empty()));
}

#[test]
fn a_save_from_before_the_turn_log_still_resumes() {
    let pl = pack("pl");
    let players = vec![
        PlayerConfig::human("Ala"),
        PlayerConfig::computer("Bot", 8, 100),
    ];
    let mut session = GameSession::new(pl.clone(), GameMode::Competitive, players, 7).unwrap();
    reply_if_due(&mut session);
    for _ in 0..2 {
        let top = session.hints(SortMode::Score).unwrap()[0].clone();
        session
            .human_move(&top.word, top.row, top.col, top.horizontal)
            .unwrap();
        reply_if_due(&mut session);
    }
    // What an older build wrote: no log, and no log length in the history.
    let mut old: serde_json::Value = serde_json::from_str(&session.to_json()).unwrap();
    old.as_object_mut().unwrap().remove("turns");
    for entry in old["history"].as_array_mut().unwrap() {
        entry.as_object_mut().unwrap().remove("turns");
    }
    let mut restored = GameSession::from_json(pl.clone(), &old.to_string()).unwrap();
    assert!(restored.state().turns.is_empty());
    restored.undo().unwrap();
    assert!(restored.state().turns.is_empty());
    // From here on it logs as any other game.
    let top = restored.hints(SortMode::Score).unwrap()[0].clone();
    restored
        .human_move(&top.word, top.row, top.col, top.horizontal)
        .unwrap();
    assert_eq!(restored.state().turns.len(), 1);
}

fn competitive(seed: u64) -> GameSession {
    let players = vec![
        PlayerConfig::human("Ala"),
        PlayerConfig::computer("Bot", 8, 100),
    ];
    let mut session = GameSession::new(pack("pl"), GameMode::Competitive, players, seed).unwrap();
    reply_if_due(&mut session);
    session
}

#[test]
fn an_equity_review_names_what_the_static_player_would_have_done() {
    let mut session = competitive(21);
    let me = 0;
    let pl = pack("pl");
    let net = pl.leave_net.as_deref().unwrap();

    // What level 9 would play here, before playing something weaker.
    let rack = session.players()[me].letters.clone();
    let opts = StaticOptions {
        use_endgame: false,
        ..StaticOptions::default()
    };
    let choice = choose_move(session.board(), &pl.lexicon, net, &rack, 90, &opts)
        .unwrap()
        .play
        .unwrap();
    let hints = session.hints(SortMode::Score).unwrap();
    let weak = hints
        .iter()
        .rev()
        .find(|h| h.word != choice.0)
        .unwrap()
        .clone();
    session
        .human_move(&weak.word, weak.row, weak.col, weak.horizontal)
        .unwrap();
    let review = session
        .state()
        .turns
        .last()
        .unwrap()
        .equity
        .clone()
        .unwrap();
    assert!(review.played < review.best, "{review:?}");
    match (&review.best_play, &review.best_exchange) {
        (Some(best), None) => assert_eq!(
            (best.word.as_str(), best.row, best.col, best.horizontal),
            (choice.0.as_str(), choice.2 .0, choice.2 .1, choice.2 .2)
        ),
        // An exchange may beat every play; then it is the one named.
        (None, Some(tiles)) => assert!(!tiles.is_empty()),
        other => panic!("a better turn must be named: {other:?}"),
    }
    reply_if_due(&mut session);

    // An exchange is valued too: by the leave it keeps.
    let two: String = session.players()[me].letters.chars().take(2).collect();
    session.exchange(&two).unwrap();
    let turn = session.state().turns.last().unwrap().clone();
    let review = turn.equity.unwrap();
    assert!(review.played <= review.best);
    assert_eq!(
        review.played == review.best,
        review.best_play.is_none() && review.best_exchange.is_none()
    );
    reply_if_due(&mut session);

    // The computer's turns are not reviewed, and no turn has an endgame yet.
    let state = session.state();
    for t in &state.turns {
        assert_eq!(t.equity.is_some(), t.player == me, "{t:?}");
        assert!(t.endgame.is_none());
    }
}

/// A competitive game played on by top-scoring moves to the human's first
/// turn with the bag empty -- for the first seed whose game gets there.
fn into_the_endgame() -> GameSession {
    for seed in 5..40 {
        let mut session = competitive(seed);
        while !session.is_over() && session.bag_remaining() > 0 {
            let top = session.hints(SortMode::Score).unwrap().first().cloned();
            match top {
                Some(h) => session
                    .human_move(&h.word, h.row, h.col, h.horizontal)
                    .map(|_| ()),
                None => session.skip(),
            }
            .unwrap();
            reply_if_due(&mut session);
        }
        if !session.is_over() {
            return session;
        }
    }
    panic!("no game reached an endgame with the human to move");
}

#[test]
fn an_endgame_turn_is_measured_against_the_search() {
    let mut session = into_the_endgame();
    let me = 0;
    assert_eq!(session.current_player_index(), me);

    // The search's own best turn loses nothing against itself.
    let pl = pack("pl");
    let rack = session.players()[me].letters.clone();
    let opponent = session.players()[1].letters.clone();
    let solved = session
        .board()
        .solve_endgame(&pl.lexicon, &rack, &opponent, INTERACTIVE_ENDGAME_NODES)
        .unwrap();
    match &solved.best {
        Some((word, _, (row, col, h), _)) => {
            session.human_move(word, *row, *col, *h).unwrap();
        }
        None => session.skip().unwrap(),
    }
    let turn = session.state().turns.last().unwrap().clone();
    assert!(
        turn.equity.is_none(),
        "with the bag empty, equity has nothing to draw on"
    );
    let review = turn.endgame.expect("an endgame review");
    assert_eq!(review.best, solved.differential.max(review.played));
    if review.exact {
        assert_eq!(review.played, review.best, "{review:?}");
    }
}

#[test]
fn the_unseen_tiles_are_the_bag_and_the_other_racks() {
    let session = competitive(9);
    let state = session.state();
    let unseen: usize = state.unseen.iter().map(|&(_, n)| n as usize).sum();
    let computer = session.players()[1].letters.chars().count();
    assert_eq!(unseen, session.bag_remaining() + computer);
    // Letter by letter: everything a full bag holds, less the board and the rack.
    let lang = session.pack().lang;
    let mine = &session.players()[0].letters;
    for &(ch, n) in &state.unseen {
        let in_bag = lang.bag().iter().filter(|&&t| t == ch).count();
        let on_board = (0..BOARD_SIZE)
            .flat_map(|r| (0..BOARD_SIZE).map(move |c| (r, c)))
            .filter(|&(r, c)| {
                let blank = state.blanks[r][c];
                if ch == lang.blank() {
                    blank
                } else {
                    !blank && state.board[r][c] == ch
                }
            })
            .count();
        let held = mine.chars().filter(|&t| t == ch).count();
        assert_eq!(n as usize, in_bag - on_board - held, "{ch}");
    }
    // A sandbox has no rack of anyone's on screen to count from.
    let players = vec![
        PlayerConfig::human("Ala"),
        PlayerConfig::computer("Bot", 8, 100),
    ];
    let sandbox = GameSession::new(pack("pl"), GameMode::Sandbox, players, 3).unwrap();
    assert!(sandbox.state().unseen.is_empty());
}

#[test]
fn a_multiplayer_game_plays_on_without_a_seat_that_resigned() {
    let pl = pack("pl");
    let bad = |players: Vec<PlayerConfig>| {
        matches!(
            GameSession::new(pl.clone(), GameMode::Multiplayer, players, 1),
            Err(GameError::InvalidPlayers(_))
        )
    };
    assert!(bad(vec![PlayerConfig::human("Ala")]));
    assert!(bad(vec![
        PlayerConfig::computer("A", 8, 100),
        PlayerConfig::computer("B", 8, 100),
    ]));

    let players = vec![
        PlayerConfig::human("Ala"),
        PlayerConfig::human("Ola"),
        PlayerConfig::computer("Bot", 8, 100),
    ];
    let mut session = GameSession::new(pl.clone(), GameMode::Multiplayer, players, 4).unwrap();
    reply_if_due(&mut session);
    // Every human plays their own rack, hinted and rated.
    for _ in 0..2 {
        let me = session.current_player_index();
        assert!(!session.players()[me].is_computer);
        let top = session.hints(SortMode::Score).unwrap()[0].clone();
        session
            .human_move(&top.word, top.row, top.col, top.horizontal)
            .unwrap();
        assert_eq!(session.state().last_move_rating, Some(100));
        reply_if_due(&mut session);
    }
    // While the game is on, nobody's racks show in the log, and only the
    // human whose turn it is has unseen tiles to count.
    let state = session.state();
    assert!(state.turns.iter().all(|t| t.rack.is_empty()));
    assert!(!state.unseen.is_empty());
    assert!(
        state.players[2].letters.is_empty(),
        "the computer's rack is hidden"
    );

    // A seat that resigns is skipped from then on.
    let quitter = session.current_player_index();
    session.resign().unwrap();
    assert!(!session.is_over());
    for _ in 0..4 {
        assert_ne!(session.current_player_index(), quitter);
        if session.computer_to_move() {
            session.computer_turn().unwrap();
        } else {
            session.skip().unwrap();
        }
        if session.is_over() {
            break;
        }
    }
    // Undo back past the resignation brings the seat back.
    while session
        .state()
        .turns
        .iter()
        .any(|t| t.action == TurnAction::Resign)
    {
        session.undo().unwrap();
    }
    assert!(!session.is_over());

    // Once the last other human resigns, the computer has the game to itself,
    // and wins it whatever the scores.
    let humans: Vec<usize> = (0..3)
        .filter(|&i| !session.players()[i].is_computer)
        .collect();
    while !session.is_over() {
        if session.computer_to_move() {
            session.computer_turn().unwrap();
        } else {
            assert!(humans.contains(&session.current_player_index()));
            session.resign().unwrap();
        }
    }
    assert_eq!(session.state().winner, Some(Winner::Player(2)));
}

#[test]
fn a_practice_play_is_checked_against_every_legal_play() {
    let session = competitive(12);
    let state = session.state();
    let rack = session.players()[0].letters.clone();
    let hints = session.hints(SortMode::Score).unwrap();
    let pl = pack("pl");
    let check =
        |w: &str, r, c, h| check_play(&pl, &state.board, &state.blanks, &rack, w, r, c, h).unwrap();
    let top = &hints[0];
    let best = check(&top.word, top.row, top.col, top.horizontal);
    assert_eq!(best.score, Some(top.score));
    assert_eq!((best.best_score, best.better), (top.score, 0));
    let equity = best.equity.unwrap();
    assert!(equity.played <= equity.best);

    let other = hints.last().unwrap();
    let weaker = check(&other.word, other.row, other.col, other.horizontal);
    assert_eq!(weaker.score, Some(other.score));
    assert!(weaker.better > 0 && weaker.better < weaker.plays);
    // Not a play at all: the right word in the wrong place.
    let r = (top.row + 7) % BOARD_SIZE;
    assert_eq!(check(&top.word, r, top.col, !top.horizontal).score, None);
}

#[test]
fn exchange_draws_before_returning_the_tiles() {
    let pack = pack("pl");
    let players = vec![
        PlayerConfig::human("Ala"),
        PlayerConfig::computer("Bot", 8, 100),
    ];
    let mut session = GameSession::new(pack, GameMode::Competitive, players, 11).unwrap();
    reply_if_due(&mut session);
    let me = session.current_player_index();
    let rack = session.players()[me].letters.clone();
    let before_bag = session.bag_remaining();
    session.exchange(&rack).unwrap();
    assert_eq!(
        session.bag_remaining(),
        before_bag,
        "an exchange keeps the bag size"
    );
    assert_eq!(session.players()[me].letters.chars().count(), 7);
    assert!(matches!(
        session.exchange("ź"),
        Err(GameError::TilesNotOnRack) | Err(GameError::WrongMode)
    ));
}

#[test]
fn a_word_must_place_a_tile_and_be_whole() {
    let pack = pack("pl");
    let players = vec![
        PlayerConfig::human("Ala"),
        PlayerConfig::computer("Bot", 8, 100),
    ];
    let mut session = GameSession::new(pack, GameMode::Sandbox, players, 3).unwrap();
    if session.players()[session.current_player_index()].is_computer {
        session.skip().unwrap();
    }
    session.human_move("kotek", 7, 5, true).unwrap();
    session.skip().unwrap(); // the computer seat
                             // Re-"playing" the word already on the board places nothing.
    assert_eq!(
        session.human_move("kotek", 7, 5, true),
        Err(GameError::NoNewTiles)
    );
    // "kot" over the start of "kotek" would leave "kotek" on the board.
    assert_eq!(
        session.human_move("kot", 7, 5, true),
        Err(GameError::NoNewTiles)
    );
    // A word starting right after "kotek" would really spell "kotek..." on
    // the board, so it must be submitted whole.
    let lexicon = session.pack().lexicon.clone();
    let tail = ["ale", "om", "aj"]
        .into_iter()
        .find(|w| lexicon.contains(w))
        .unwrap();
    assert_eq!(
        session.human_move(tail, 7, 10, true),
        Err(GameError::WordNotWhole)
    );
}

#[test]
fn a_benchmark_folds_every_game_into_the_stats() {
    let pack = pack("pl");
    let players = vec![
        PlayerConfig::computer("Mocny", 8, 100),
        PlayerConfig::computer("Słaby", 2, 100),
    ];
    let control = BenchmarkControl::default();
    let result = run_benchmark(&pack, &players, 4, 99, &control).unwrap();
    assert_eq!(result.games_played, 4);
    assert_eq!(control.games_done(), 4);
    for s in &result.player_stats {
        assert_eq!(s.games_played, 4);
        assert!(s.low_score <= s.high_score);
    }
    let decided: usize = result.player_stats.iter().map(|s| s.wins).sum();
    let tied = result.player_stats.iter().map(|s| s.ties).max().unwrap();
    assert_eq!(decided + tied, 4, "every game has one winner or a tie");
    let best = result.best_game.as_ref().expect("a best game");
    assert!(!best.moves.is_empty());
    let last = best.moves.last().unwrap();
    assert_eq!(
        best.final_players.iter().map(|p| p.score).max().unwrap(),
        best.winner_score
    );
    assert!(last.scores_after.len() == 2);
    // Same seed, same run.
    let again = run_benchmark(&pack, &players, 4, 99, &BenchmarkControl::default()).unwrap();
    assert_eq!(again.player_stats, result.player_stats);
}

#[test]
fn a_scan_trusts_the_previous_board_over_the_photo() {
    use crate::scan::{assemble, TileReading};
    let pack = pack("pl");
    let read = |ch: char, alts: &[(char, f32)]| TileReading {
        letter: Some(ch),
        confidence: 0.9,
        alternatives: alts.to_vec(),
    };
    // "kot" across the centre, confirmed last time.
    let mut prior = [['-'; BOARD_SIZE]; BOARD_SIZE];
    for (i, ch) in "kot".chars().enumerate() {
        prior[7][6 + i] = ch;
    }
    let mut readings = HashMap::new();
    readings.insert((7, 6), read('k', &[]));
    // The photo misreads the old 'o' as 'a': history wins, the fresh read is
    // kept as the first alternative.
    readings.insert((7, 7), read('a', &[('e', 0.5)]));
    readings.insert((7, 8), read('t', &[]));
    // A new tile making "kota", misread as 'ź' ("kotź" is no word) with
    // 'a' as a real second guess.
    readings.insert((7, 9), read('ź', &[('a', 0.4), ('e', 0.05)]));
    let cells = assemble(&pack.lexicon, &readings, Some(&prior));
    assert_eq!(cells[7][7].letter, 'o');
    assert!(cells[7][7].carried_over);
    assert_eq!(cells[7][7].alternatives.first(), Some(&'a'));
    assert_eq!(cells[7][9].letter, 'a', "corrected by the dictionary");
    assert!(!cells[7][9].flagged);
    assert!(
        !cells[7][9].alternatives.contains(&'e'),
        "a 5% guess is noise, not an alternative"
    );
}

#[test]
fn a_simulating_bot_replays_from_its_seed() {
    let pack = pack("pl");
    let play = || {
        let seats = vec![
            PlayerConfig::computer("Analityk", 10, 100),
            PlayerConfig::computer("Słabszy", 10, 60),
        ];
        let mut session = GameSession::new(pack.clone(), GameMode::SandboxAuto, seats, 5).unwrap();
        for _ in 0..4 {
            session.next_auto_move().unwrap();
        }
        // The game as played -- board, racks, scores, last move. (`to_json`
        // would also hold the engine board's own bag, which a session never
        // deals from and which is seeded from the clock.)
        session.state()
    };
    assert_eq!(play(), play(), "a seeded game must replay move for move");
}
