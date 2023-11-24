mod consts;

use consts::*;
use pyo3::ffi::newfunc;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result};
use std::fs;
use std::hash::{Hash, Hasher};

static mut NEXT_ID: i32 = 0;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Word {
    is_vertical: bool,
    x: u8,
    y: u8,
    word: String,
    score: i16,
    av_letters: Vec<char>,
}

impl Word {
    fn new() -> Self {
        Word {
            is_vertical: false,
            x: 0,
            y: 0,
            word: "".to_string(),
            score: 0,
            av_letters: vec![],
        }
    }
}

impl Ord for Word {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

impl PartialOrd for Word {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Eq)]
struct Node {
    children: HashMap<char, Node>,
    is_terminal: bool,
    id: i32,
}

impl Display for Node {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut out: Vec<String> = Vec::new();
        if self.is_terminal {
            out.push("1".to_string());
        } else {
            out.push("0".to_string());
        }
        for (key, val) in &self.children {
            out.push(key.to_string());
            out.push(val.id.to_string());
        }

        write!(f, "{}", out.join("_"))
    }
}

impl Node {
    fn new() -> Self {
        unsafe { NEXT_ID += 1 };
        Node {
            children: HashMap::new(),
            is_terminal: false,
            id: unsafe { NEXT_ID - 1 },
        }
    }
    fn __repr__(&self) -> String {
        format!("{}", self)
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.__repr__().hash(state);
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        self.__repr__() == other.__repr__()
    }
}

#[derive(Debug, Clone)]
struct Game {
    dawg: Node,
    board: Vec<Vec<char>>,
    tile_bag: Vec<char>,
    end: bool,
}

impl Display for Game {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let pretty_board: String = self
            .board
            .iter()
            .map(|row| {
                row.iter()
                    .map(|chr| chr.to_string().to_ascii_uppercase())
                    .collect::<Vec<String>>()
                    .join(" ")
            })
            .collect::<Vec<String>>()
            .join("\n");
        write!(f, "{}", pretty_board)
    }
}

impl Game {
    fn new(dawg: Node) -> Self {
        Game {
            dawg,
            board: vec![vec!['-'; 15]; 15],
            tile_bag: TILE_BAG.to_vec(),
            end: false,
        }
    }

    fn __repr__(&self) -> String {
        format!("{}", self)
    }

    fn insert_word(&mut self, word: &mut Word) {
        if word.is_vertical {
            self.board = (0..15)
                .map(|col| (0..15).map(|row| self.board[row][col]).collect())
                .collect();
            (word.x, word.y) = (word.y, word.x);
        }
        for i in 0..(word.word.chars().count() as u8) {
            self.board[word.x as usize][(word.y + i) as usize] = word
                .word
                .chars()
                .nth(i as usize)
                .expect("no letter in insert word function!");
        }
        if word.is_vertical {
            self.board = (0..15)
                .map(|col| (0..15).map(|row| self.board[row][col]).collect())
                .collect();
            (word.x, word.y) = (word.y, word.x)
        }
    }

    fn give_new_letters(&mut self, letters: &Vec<char>) -> Vec<char> {
        let mut rng = rand::thread_rng();
        let new_letters: Vec<char> = (0..min(7 - letters.len(), self.tile_bag.len()))
            .filter_map(|_| self.tile_bag.choose(&mut rng))
            .cloned()
            .collect();
        for letter in new_letters.iter() {
            let index = self
                .tile_bag
                .iter()
                .position(|x| *x == *letter)
                .expect("letter not in new letters in give_new_letters function");
            self.tile_bag.remove(index);
        }
        new_letters
    }
}

struct Player<'a> {
    letters: Vec<char>,
    score: i16,
    game: &'a mut Game,
}

impl<'a> Player<'a> {
    fn new(game: &'a mut Game) -> Self {
        Player {
            letters: game.give_new_letters(&Vec::new()),
            score: 0,
            game,
        }
    }

    fn exchange_letters(&mut self, n: i16) {
        self.letters.shuffle(&mut rand::thread_rng());

        for _ in 0..(n.min(self.letters.len() as i16)) {
            let letter: char = self.letters.pop().expect("List is empty?");
            self.game.tile_bag.push(letter);
        }
        self.get_new_letters();
    }

    fn get_new_letters(&mut self) {
        let new_letters = self.game.give_new_letters(&self.letters);
        self.letters.extend(new_letters);
    }

    fn validate_word(&self, node: &Node, word: String, x: i16) -> String {
        if x == word.chars().count() as i16 {
            if node.is_terminal {
                return word;
            }
            return "".to_string();
        }
        node.children
            .get(&word.chars().nth(x as usize).unwrap())
            .and_then(|child| {
                return Some(self.validate_word(child, word, x + 1));
            })
            .unwrap_or_else(|| {
                return "".to_string();
            })
    }

    fn check_crossword(&self, column: Vec<char>, new_letter: char, y: u8, x: u8) -> (bool, i16) {
        let mut score: i16 = 0;
        let pattern = Regex::new(r"\w+").expect("");
        let str_column: String = column.iter().collect();

        for result in pattern.find_iter(&str_column) {
            if result.start() as u8 <= y && y <= result.end() as u8 {
                if self.validate_word(
                    &self.game.dawg,
                    column[result.start()..result.end()].iter().collect(),
                    0,
                ) != ""
                {
                    for letter in &column[result.start()..result.end()] {
                        score += LETTER_POINTS.get(letter).expect("letter not in hashmap");
                    }
                    if BONUSES.contains_key(&[y, x]) {
                        score += LETTER_POINTS[&new_letter] * (BONUSES[&[y, x]].0 as i16 - 1);
                        score *= BONUSES[&[y, x]].1 as i16;
                    }
                    return (true, score);
                }
            }
        }
        return (false, 0);
    }

    fn find_first_words(
        &self,
        node: &Node,
        av_letters: &Vec<char>,
        mut best_word: Word,
        mut can_be: bool,
        word: Word,
        points: [i16; 2],
        x: u8,
    ) -> Word {
        if node.is_terminal && can_be {
            let mut new_word = word.clone();
            new_word.x = 7;
            new_word.y = 7;
            new_word.score = points[0] * points[1];
            new_word.av_letters = av_letters.to_vec();

            if av_letters.len() == 7 {
                new_word.score += 50;
            }

            if best_word.score != 0 {
                best_word = max(best_word, new_word);
            } else {
                best_word = new_word;
            }
        }

        if x == 15 {
            return best_word;
        }

        for (letter, child) in node.children.iter() {
            if !av_letters.contains(&letter) {
                continue;
            }

            if x == 7 {
                can_be = true;
            }

            let bonus = BONUSES
                .get(&[7, x])
                .map(|bon| *bon)
                .unwrap_or_else(|| (1, 1));

            let mut new_av_letters = av_letters.clone();
            new_av_letters.remove(
                av_letters
                    .iter()
                    .position(|value| value == letter)
                    .expect("no letter in av_letters"),
            );

            let mut new_word = word.clone();
            new_word.word += letter.to_string().as_str();

            let mut new_points = points.clone();
            new_points[0] += LETTER_POINTS[&letter] * bonus.0 as i16;
            new_points[1] *= bonus.1 as i16;

            best_word = self.find_first_words(
                child,
                &new_av_letters,
                best_word,
                can_be,
                new_word,
                new_points,
                x + 1,
            );
        }

        if word.word == "" {
            best_word =
                self.find_first_words(node, av_letters, best_word, can_be, word, points, x + 1);
        }

        best_word
    }

    fn place_best_first_word(&mut self) -> Word {
        let mut best_word: Word = self.find_first_words(
            &self.game.dawg,
            &self.letters,
            Word::new(),
            false,
            Word::new(),
            [0, 1],
            0,
        );

        if best_word.score != 0 {
            self.game.insert_word(&mut best_word);
            self.letters = best_word.av_letters.clone();
            self.score += best_word.score;
            self.get_new_letters();
        }

        best_word
    }

    fn find_words(&self, node: &Node, av_letters: &Vec<char>, best_word: Word) -> Word {
        return best_word;
    }

    fn place_best_word(&self) -> Word {
        let best_word = self.find_words(&self.game.dawg, &self.letters, Word::new());

        return best_word;
    }

    fn make_move(&mut self, first: bool) -> Word {
        let word;
        if first {
            word = self.place_best_first_word();
        } else {
            word = self.place_best_word();
        }
        if word.score == 0 {
            if self.game.tile_bag.is_empty() {
                self.game.end = true;
                return Word::new();
            } else {
                self.exchange_letters(2);
            }
        }
        word
    }
}

#[pyfunction]
fn play_game() {
    let data = fs::read_to_string("./dawg.json").expect("Unable to read file");
    let node: Node = serde_json::from_str(&data).expect("JSON does not have correct format.");
    println!("{}", node);
    let mut game: Game = Game::new(node.clone());
    let mut player1: Player = Player::new(&mut game);
    println!("{:?}", player1.letters);
    println!("{:?}", player1.make_move(true));
    println!("{}", game);
    // println!("{}", game);
}

#[derive(Serialize, Deserialize)]
struct Person {
    name: String,
    age: u8,
    phones: Vec<String>,
}

#[pymodule]
fn scrablozaur(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(play_game, m)?)?;
    Ok(())
}
