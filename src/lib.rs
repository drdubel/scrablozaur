use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Node (używany tylko podczas budowania)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Node {
    is_terminal: bool,
    children: HashMap<char, u32>,
}

struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }
    fn alloc(&mut self) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(Node::default());
        id
    }
    fn node(&self, id: u32) -> &Node {
        &self.nodes[id as usize]
    }
    fn node_mut(&mut self, id: u32) -> &mut Node {
        &mut self.nodes[id as usize]
    }
}

// ---------------------------------------------------------------------------
// Dawg – spłaszczona, read-only reprezentacja załadowana z pliku.
// ---------------------------------------------------------------------------

struct Dawg {
    data: Vec<u8>,
    root: u32,
    offset_table: Vec<usize>,
}

impl Dawg {
    fn load(path: &str) -> io::Result<Self> {
        let data = fs::read(path)?;
        if data.len() < 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "plik za krótki"));
        }
        let root = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let node_count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

        let mut offset_table = Vec::with_capacity(node_count);
        let mut pos = 8usize;
        for _ in 0..node_count {
            offset_table.push(pos);
            pos += 1;
            let children_count =
                u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            pos += children_count * 8;
        }

        Ok(Self {
            data,
            root,
            offset_table,
        })
    }

    #[inline]
    fn node_is_terminal(&self, id: u32) -> bool {
        self.data[self.offset_table[id as usize]] != 0
    }

    #[inline]
    fn node_children_count(&self, id: u32) -> usize {
        let base = self.offset_table[id as usize] + 1;
        u32::from_le_bytes(self.data[base..base + 4].try_into().unwrap()) as usize
    }

    #[inline]
    fn node_child(&self, id: u32, i: usize) -> (char, u32) {
        let base = self.offset_table[id as usize] + 1 + 4 + i * 8;
        let cp = u32::from_le_bytes(self.data[base..base + 4].try_into().unwrap());
        let cid = u32::from_le_bytes(self.data[base + 4..base + 8].try_into().unwrap());
        (char::from_u32(cp).unwrap(), cid)
    }

    #[inline]
    fn find_child(&self, id: u32, c: char) -> Option<u32> {
        let n = self.node_children_count(id);
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (mc, mid_id) = self.node_child(id, mid);
            match mc.cmp(&c) {
                std::cmp::Ordering::Equal => return Some(mid_id),
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }

    fn contains(&self, word: &str) -> bool {
        let mut curr = self.root;
        for c in word.chars() {
            match self.find_child(curr, c) {
                Some(next) => curr = next,
                None => return false,
            }
        }
        self.node_is_terminal(curr)
    }

    fn node_count(&self) -> usize {
        self.offset_table.len()
    }

    /// Wzorzec:
    ///   - zwykła litera – musi wystąpić dokładnie na tej pozycji
    ///   - '-'           – dokładnie jedna litera pobrana z worka
    ///   - '*'           – zero lub więcej liter pobranych z worka
    ///
    /// Worek z liczeniem – "aab" pozwala użyć 'a' dwa razy i 'b' raz łącznie.
    fn match_pattern(
        &self,
        pattern: &[char],
        pat_pos: usize,
        node_id: u32,
        letters: &mut Vec<char>,
        mandatory_slots: usize,
        results: &mut Vec<String>,
        current: &mut String,
    ) {
        if pat_pos == pattern.len() {
            if self.node_is_terminal(node_id) {
                results.push(current.clone());
            }
            return;
        }

        // Pomocnik: próbuje wstawić literę c (lub '?' jako wildcard) na bieżącą pozycję.
        // Wywołuje rekurencję z pat_pos+1 lub tym samym pat_pos (dla '*').
        // Zwraca true jeśli udało się wziąć literę z worka.

        match pattern[pat_pos] {
            '-' => {
                let n = self.node_children_count(node_id);
                for i in 0..n {
                    let (c, child_id) = self.node_child(node_id, i);
                    // próbuj konkretną literę
                    if let Some(pos) = letters.iter().position(|&l| l == c) {
                        letters.swap_remove(pos);
                        current.push(c);
                        self.match_pattern(
                            pattern,
                            pat_pos + 1,
                            child_id,
                            letters,
                            mandatory_slots - 1,
                            results,
                            current,
                        );
                        current.pop();
                        letters.push(c);
                    }
                    // próbuj '?' jako wildcard (jeśli jest w worku)
                    if let Some(pos) = letters.iter().position(|&l| l == '?') {
                        letters.swap_remove(pos);
                        current.push(c);
                        self.match_pattern(
                            pattern,
                            pat_pos + 1,
                            child_id,
                            letters,
                            mandatory_slots - 1,
                            results,
                            current,
                        );
                        current.pop();
                        letters.push('?');
                    }
                }
            }
            '*' => {
                // Optymalizacja: '*' może zużyć co najwyżej (worek - mandatory_slots) liter.
                // Jeśli nie ma już liter ponad wymagane minimum, '*' może tylko
                // dopasować zero liter.
                let available_for_star = letters.len().saturating_sub(mandatory_slots);

                // zero liter – przechodzimy dalej we wzorcu, zostajemy w węźle
                self.match_pattern(
                    pattern,
                    pat_pos + 1,
                    node_id,
                    letters,
                    mandatory_slots,
                    results,
                    current,
                );

                if available_for_star > 0 {
                    // jedna lub więcej liter – schodzimy do dziecka, zostajemy przy '*'
                    let n = self.node_children_count(node_id);
                    for i in 0..n {
                        let (c, child_id) = self.node_child(node_id, i);
                        // konkretna litera
                        if let Some(pos) = letters.iter().position(|&l| l == c) {
                            letters.swap_remove(pos);
                            current.push(c);
                            self.match_pattern(
                                pattern,
                                pat_pos,
                                child_id,
                                letters,
                                mandatory_slots,
                                results,
                                current,
                            );
                            current.pop();
                            letters.push(c);
                        }
                        // '?' jako wildcard
                        if let Some(pos) = letters.iter().position(|&l| l == '?') {
                            letters.swap_remove(pos);
                            current.push(c);
                            self.match_pattern(
                                pattern,
                                pat_pos,
                                child_id,
                                letters,
                                mandatory_slots,
                                results,
                                current,
                            );
                            current.pop();
                            letters.push('?');
                        }
                    }
                }
            }
            fixed_char => {
                if let Some(child_id) = self.find_child(node_id, fixed_char) {
                    current.push(fixed_char);
                    self.match_pattern(
                        pattern,
                        pat_pos + 1,
                        child_id,
                        letters,
                        mandatory_slots,
                        results,
                        current,
                    );
                    current.pop();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DawgPy – wrapper eksponowany do Pythona
// ---------------------------------------------------------------------------

#[pyclass(name = "Dawg")]
struct DawgPy {
    inner: Dawg,
}

#[pymethods]
impl DawgPy {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Dawg::load(path)
            .map(|inner| DawgPy { inner })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    fn contains(&self, word: &str) -> bool {
        self.inner.contains(word)
    }

    fn __contains__(&self, word: &str) -> bool {
        self.inner.contains(word)
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Szuka słów pasujących do wzorca.
    ///
    /// Przykład:
    ///   d.search("l--y", "oadn")
    ///   → ["lody", "lady", ...] – słowa gdzie l.*y, środkowe litery z "oadn"
    ///
    /// Wzorzec:
    ///   - zwykła litera = musi być na tej pozycji
    ///   - '-'           = dokładnie jedna litera z zestawu (z zachowaniem liczności)
    fn search(&self, pattern: &str, letters: &str) -> Vec<String> {
        let pattern_chars: Vec<char> = pattern.chars().collect();
        let mut letters_bag: Vec<char> = letters.chars().collect();

        // Oblicz ile myślników jest we wzorcu – tyle liter z worka musi być
        // zarezerwowane i nie może ich zużyć '*'.
        let mandatory_slots = pattern_chars.iter().filter(|&&c| c == '-').count();

        let mut results = Vec::new();
        let mut current = String::with_capacity(pattern.len());

        self.inner.match_pattern(
            &pattern_chars,
            0,
            self.inner.root,
            &mut letters_bag,
            mandatory_slots,
            &mut results,
            &mut current,
        );

        // '?' może dopasować tę samą literę w DAWG-u na wiele sposobów
        // (raz jako konkretna litera z worka, raz jako '?') – deduplikuj.
        results.sort_unstable();
        results.dedup();
        results
    }
}

#[pyclass(name = "Board")]
struct Board {
    board: [[char; 15]; 15],
}

#[pymethods]
impl Board {
    #[new]
    fn new(board: Vec<Vec<String>>) -> PyResult<Self> {
        if board.len() != 15 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Plansza musi mieć dokładnie 15 wierszy",
            ));
        }

        let mut result = [['-'; 15]; 15];
        for (r, row) in board.iter().enumerate() {
            if row.len() != 15 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Każdy wiersz musi mieć dokładnie 15 kolumn",
                ));
            }
            for (c, cell) in row.iter().enumerate() {
                let mut chars = cell.chars();
                let ch = chars
                    .next()
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Pusta komórka"))?;
                if chars.next().is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Komórka musi zawierać dokładnie jeden znak",
                    ));
                }
                result[r][c] = ch;
            }
        }

        Ok(Board { board: result })
    }

    fn calculate_word_points(&self, word: &str, row: usize, col: usize, horizontal: bool) -> u32 {
        let mut word_mul = 1;

        let letter_points = |c: char| match c.to_ascii_uppercase() {
            'A' | 'E' | 'I' | 'O' | 'Z' | 'W' | 'N' | 'S' | 'R' => 1,
            'D' | 'Y' | 'C' | 'K' | 'L' | 'M' | 'P' | 'T' => 2,
            'B' | 'G' | 'H' | 'J' | 'Ł' | 'U' => 3,
            'Ą' | 'Ę' | 'F' | 'Ó' | 'Ś' | 'Ż' => 4,
            'Ć' => 6,
            'Ń' => 7,
            'Ź' => 9,
            _ => 0,
        };

        // (letter_mul, word_mul)
        static BONUSES: &[((u8, u8), (u8, u8)); 18] = &[
            ((0, 0), (1, 3)),
            ((0, 3), (2, 1)),
            ((0, 7), (1, 3)),
            ((1, 1), (1, 2)),
            ((1, 5), (3, 1)),
            ((2, 2), (1, 2)),
            ((2, 6), (2, 1)),
            ((3, 0), (2, 1)),
            ((3, 3), (1, 2)),
            ((3, 7), (2, 1)),
            ((4, 4), (1, 2)),
            ((5, 1), (3, 1)),
            ((5, 5), (3, 1)),
            ((6, 2), (2, 1)),
            ((6, 6), (2, 1)),
            ((7, 0), (1, 3)),
            ((7, 3), (2, 1)),
            ((7, 7), (1, 2)),
        ];

        fn get_bonus(r: u8, c: u8) -> (u8, u8) {
            let key = (r, c);
            BONUSES
                .iter()
                .find(|(pos, _)| *pos == key)
                .map(|(_, bonus)| *bonus)
                .unwrap_or((1, 1))
        }

        let mut total_points = 0;
        for (i, ch) in word.chars().enumerate() {
            let r = if horizontal { row } else { row + i };
            let c = if horizontal { col + i } else { col };

            let (r2, c2) = ((r as u8).min(14 - r as u8), (c as u8).min(14 - c as u8));

            if r >= 15 || c >= 15 {
                panic!("Słowo wykracza poza planszę");
            }
            if self.board[r][c] == '-' {
                total_points += letter_points(ch) * get_bonus(r2, c2).0 as u32;
                word_mul *= get_bonus(r2, c2).1 as u32;
            } else {
                total_points += letter_points(ch);
            }
        }
        total_points * word_mul
    }
}

// ---------------------------------------------------------------------------
// Budowanie DAWG-a
// ---------------------------------------------------------------------------

fn node_key(arena: &Arena, id: u32) -> String {
    let node = arena.node(id);
    let mut out = String::new();
    out.push(if node.is_terminal { '1' } else { '0' });
    let mut pairs: Vec<(char, u32)> = node.children.iter().map(|(&c, &id)| (c, id)).collect();
    pairs.sort_unstable_by_key(|&(c, _)| c);
    for (c, child_id) in pairs {
        out.push('_');
        out.push(c);
        out.push('_');
        out.push_str(&child_id.to_string());
    }
    out
}

fn prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

fn minimize(
    arena: &mut Arena,
    pref_len: usize,
    minimized: &mut HashMap<String, u32>,
    stack: &mut Vec<(u32, char, u32)>,
) {
    let pop_count = stack.len() - pref_len;
    for _ in 0..pop_count {
        let (parent_id, letter, child_id) = stack.pop().unwrap();
        let key = node_key(arena, child_id);
        let canonical_id = if let Some(&existing) = minimized.get(&key) {
            existing
        } else {
            minimized.insert(key, child_id);
            child_id
        };
        arena
            .node_mut(parent_id)
            .children
            .insert(letter, canonical_id);
    }
}

fn build_dawg(words: &[&str]) -> (Arena, u32, usize) {
    let mut arena = Arena::new();
    let root = arena.alloc();
    let mut minimized: HashMap<String, u32> = HashMap::new();
    minimized.insert(node_key(&arena, root), root);
    let mut stack: Vec<(u32, char, u32)> = Vec::new();
    let mut curr = root;
    let mut prev_word = "";

    let total = words.len();
    let report_every = (total / 40).max(1);

    for (i, &word) in words.iter().enumerate() {
        if i % report_every == 0 {
            eprint!("\r  budowanie: {}/{}", i, total);
            let _ = io::stderr().flush();
        }
        let pref = prefix_len(prev_word, word);
        if !stack.is_empty() {
            minimize(&mut arena, pref, &mut minimized, &mut stack);
            curr = stack.last().map(|&(_, _, child)| child).unwrap_or(root);
        }
        for c in word.chars().skip(pref) {
            let child = arena.alloc();
            arena.node_mut(curr).children.insert(c, child);
            stack.push((curr, c, child));
            curr = child;
        }
        arena.node_mut(curr).is_terminal = true;
        prev_word = word;
    }
    minimize(&mut arena, 0, &mut minimized, &mut stack);
    eprintln!("\r  budowanie: {}/{}", total, total);

    (arena, root, minimized.len())
}

fn serialize(arena: &Arena, root: u32) -> Vec<u8> {
    let n = arena.nodes.len();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&root.to_le_bytes());
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    for id in 0..n as u32 {
        let node = arena.node(id);
        buf.push(node.is_terminal as u8);
        let mut children: Vec<(char, u32)> =
            node.children.iter().map(|(&c, &cid)| (c, cid)).collect();
        children.sort_unstable_by_key(|&(c, _)| c);
        buf.extend_from_slice(&(children.len() as u32).to_le_bytes());
        for (c, cid) in children {
            buf.extend_from_slice(&(c as u32).to_le_bytes());
            buf.extend_from_slice(&cid.to_le_bytes());
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn usage(prog: &str) {
    eprintln!(
        "Użycie:
  {prog} build  <słowa.txt>  <dawg.bin>
  {prog} lookup <dawg.bin>   <słowo>
  {prog} bench  <dawg.bin>   <słowa.txt>"
    );
}

fn cmd_build(words_path: &str, dawg_path: &str) -> io::Result<()> {
    eprintln!("Wczytywanie '{words_path}'…");
    let text = fs::read_to_string(words_path)?;
    let mut words: Vec<&str> = text.split_whitespace().collect();
    words.sort_unstable();
    words.dedup();
    eprintln!("  {} unikalnych słów", words.len());

    let t0 = Instant::now();
    let (arena, root, node_count) = build_dawg(&words);
    eprintln!(
        "  gotowe w {:.2?}  │  {} węzłów po minimalizacji",
        t0.elapsed(),
        node_count
    );

    let data = serialize(&arena, root);
    {
        let file = fs::File::create(dawg_path)?;
        let mut bw = BufWriter::new(file);
        bw.write_all(&data)?;
    }
    eprintln!(
        "  {:.3} MiB → '{dawg_path}'",
        data.len() as f64 / (1 << 20) as f64
    );
    Ok(())
}

fn cmd_lookup(dawg_path: &str, word: &str) -> io::Result<()> {
    let dawg = Dawg::load(dawg_path)?;
    let t0 = Instant::now();
    let found = dawg.contains(word);
    let elapsed = t0.elapsed();
    if found {
        println!("✓  \"{word}\" – JEST w słowniku  ({elapsed:.2?})");
    } else {
        println!("✗  \"{word}\" – NIE MA w słowniku  ({elapsed:.2?})");
    }
    Ok(())
}

fn cmd_bench(dawg_path: &str, words_path: &str) -> io::Result<()> {
    let dawg = Dawg::load(dawg_path)?;
    let text = fs::read_to_string(words_path)?;
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();

    let warmup = (n / 10).max(1000).min(n);
    for w in &words[..warmup] {
        std::hint::black_box(dawg.contains(w));
    }

    const PASSES: u32 = 5;
    let mut found = 0usize;
    let t0 = Instant::now();
    for _ in 0..PASSES {
        for w in &words {
            if dawg.contains(w) {
                found += 1;
            }
        }
    }
    let elapsed = t0.elapsed();
    let total = n * PASSES as usize;
    let secs = elapsed.as_secs_f64();

    println!("\nWyniki ({PASSES} × {n} = {total} lookupów):");
    println!("  czas łączny  : {elapsed:.3?}");
    println!("  przepustowość: {:.0} lookupów/s", total as f64 / secs);
    println!("  czas/lookup  : {:.1} ns", secs * 1e9 / total as f64);
    println!(
        "  trafień      : {found}/{total} ({:.1}%)",
        100.0 * found as f64 / total as f64
    );
    Ok(())
}

pub fn main_cli() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") if args.len() == 4 => cmd_build(&args[2], &args[3]),
        Some("lookup") if args.len() == 4 => cmd_lookup(&args[2], &args[3]),
        Some("bench") if args.len() == 4 => cmd_bench(&args[2], &args[3]),
        _ => {
            usage(&args[0]);
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Moduł Python
// ---------------------------------------------------------------------------

#[pymodule]
fn scrablozaur(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DawgPy>()?;
    m.add_class::<Board>()?;
    Ok(())
}
