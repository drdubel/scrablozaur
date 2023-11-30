mod consts;

use consts::*;
use lazy_static::lazy_static;
use rand::seq::SliceRandom;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result};
use std::hash::{Hash, Hasher};

const DATA: &str = include_str!("./dawg.json");

lazy_static! {
    static ref NODE: Node =
        serde_json::from_str(&DATA).expect("JSON does not have correct format.");
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Word {
    is_vertical: bool,
    x: usize,
    y: usize,
    word: String,
    score: u16,
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
                    .map(|chr| chr.to_string().to_uppercase())
                    .collect::<Vec<String>>()
                    .join(" ")
            })
            .collect::<Vec<String>>()
            .join("\n");
        write!(f, "{}", pretty_board)
    }
}

impl Game {
    fn new() -> Self {
        Game {
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
        for i in 0..word.word.chars().count() {
            self.board[word.y][word.x + i] = word
                .word
                .chars()
                .nth(i)
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
        let new_letters: Vec<char> = self
            .tile_bag
            .choose_multiple(&mut rng, min(7 - letters.len(), self.tile_bag.len()))
            .cloned()
            .collect();
        for letter in new_letters.iter() {
            let index = self
                .tile_bag
                .iter()
                .position(|x| *x == *letter)
                .expect("letter not in tile_bag in give_new_letters function");
            self.tile_bag.remove(index);
        }
        new_letters
    }
}

struct Player<'a> {
    letters: Vec<char>,
    score: u16,
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

    fn exchange_letters(&mut self, n: u16) {
        self.letters.shuffle(&mut rand::thread_rng());

        for _ in 0..(n.min(self.letters.len() as u16)) {
            let letter: char = self.letters.pop().expect("List is empty?");
            self.game.tile_bag.push(letter);
        }
        self.get_new_letters();
    }

    fn get_new_letters(&mut self) {
        let new_letters = self.game.give_new_letters(&self.letters);
        self.letters.extend(new_letters);
    }

    fn validate_word(&self, node: &Node, word: String, x: usize) -> String {
        if x == word.chars().count() {
            if node.is_terminal {
                return word;
            }
            return "".to_string();
        }
        node.children
            .get(&word.chars().nth(x).unwrap())
            .and_then(|child| {
                return Some(self.validate_word(child, word, x + 1));
            })
            .unwrap_or_else(|| {
                return "".to_string();
            })
    }

    fn check_crossword(
        &self,
        column: Vec<char>,
        new_letter: char,
        y: usize,
        x: usize,
    ) -> (bool, u16) {
        let mut score: u16 = 0;
        let pattern = Regex::new(r"\w+").unwrap();
        let str_column: String = column.iter().collect();

        for result in pattern.find_iter(&str_column) {
            if result.start() <= y
                && y <= result.end() - (result.as_str().len() - result.as_str().chars().count())
            {
                // println!("{} {} {}", result.as_str(), result.start(), result.end());
                if self.validate_word(&NODE, result.as_str().to_string(), 0) != "" {
                    for letter in result.as_str().chars() {
                        score += LETTER_POINTS.get(&letter).expect("letter not in hashmap");
                    }
                    if BONUSES.contains_key(&[y, x]) {
                        score += LETTER_POINTS[&new_letter] * (BONUSES[&[y, x]].0 as u16 - 1);
                        score *= BONUSES[&[y, x]].1 as u16;
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
        word: String,
        points: [u16; 2],
        x: usize,
    ) -> Word {
        if node.is_terminal && can_be {
            let mut new_word = Word::new();
            new_word.word = word.clone();
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

            best_word = self.find_first_words(
                child,
                &new_av_letters,
                best_word,
                can_be,
                word.clone() + letter.to_string().as_str(),
                [
                    points[0] + LETTER_POINTS[&letter] * bonus.0 as u16,
                    points[1] * bonus.1 as u16,
                ],
                x + 1,
            );
        }

        if word == "" {
            best_word =
                self.find_first_words(node, av_letters, best_word, false, word, points, x + 1);
        }

        best_word
    }

    fn place_best_first_word(&mut self) -> Word {
        let mut best_word: Word = self.find_first_words(
            &NODE,
            &self.letters,
            Word::new(),
            false,
            "".to_string(),
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

    fn find_words(
        &self,
        node: &Node,
        av_letters: &Vec<char>,
        mut best_word: Word,
        y: usize,
        is_vertical: bool,
        word: String,
        can_be: [bool; 2],
        points: [u16; 3],
        x: usize,
    ) -> Word {
        if node.is_terminal
            && can_be[0]
            && can_be[1]
            && (x == 15
                || (x <= 14 && self.game.board[y][x] == '-')
                || (x <= 13 && self.game.board[y][x] == '-' && self.game.board[y][x + 1] == '-'))
        {
            let mut new_word = Word::new();

            new_word.is_vertical = is_vertical;
            new_word.word = word.clone();
            new_word.x = x - new_word.word.chars().count();
            new_word.y = y;
            if is_vertical {
                (new_word.x, new_word.y) = (new_word.y, new_word.x);
            }

            new_word.score = points[0] * points[2] + points[1];
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

        if self.game.board[y][x] != '-'
            && node.children.contains_key(&self.game.board[y][x])
            && !(word == "" && self.game.board[y][x] != '-')
        {
            best_word = self.find_words(
                &node.children[&self.game.board[y][x]],
                av_letters,
                best_word.clone(),
                y,
                is_vertical,
                word.clone() + self.game.board[y][x].to_string().as_str(),
                [true, can_be[1]],
                [
                    points[0] + LETTER_POINTS[&self.game.board[y][x]],
                    points[1],
                    points[2],
                ],
                x,
            )
        } else if self.game.board[y][x] == '-'
            && !(word == "" && x > 0 && self.game.board[y][x - 1] != '-')
        {
            let mut cross_points: u16;
            let mut is_cross_word: bool;

            let columns: Vec<Vec<char>> = (0..15)
                .map(|col| (0..15).map(|row| self.game.board[row][col]).collect())
                .collect();

            for (letter, child) in node.children.iter() {
                if !av_letters.contains(&letter) {
                    continue;
                }

                cross_points = 0;
                is_cross_word = false;

                if (y > 0 && self.game.board[y - 1][x] != '-')
                    || (y < 14 && self.game.board[y + 1][x] != '-')
                {
                    let mut column: Vec<char> = columns[x].clone();
                    column[y] = *letter;

                    (is_cross_word, cross_points) = self.check_crossword(column, *letter, y, x);

                    if !is_cross_word {
                        continue;
                    }
                }

                let bonus = BONUSES
                    .get(&[y, x])
                    .map(|bon| *bon)
                    .unwrap_or_else(|| (1, 1));

                let mut new_av_letters = av_letters.clone();
                new_av_letters.remove(
                    av_letters
                        .iter()
                        .position(|value| value == letter)
                        .expect("no letter in av_letters"),
                );

                let mut new_can_be = [can_be[0], true];
                if is_cross_word {
                    new_can_be[0] = true;
                }

                best_word = self.find_words(
                    child,
                    &new_av_letters,
                    best_word.clone(),
                    y,
                    is_vertical,
                    word.clone() + &letter.to_string(),
                    new_can_be,
                    [
                        points[0] + LETTER_POINTS[&letter] * bonus.0 as u16,
                        points[1] + cross_points,
                        points[2] * bonus.1 as u16,
                    ],
                    x + 1,
                );
            }
        }

        if word == "" {
            best_word = self.find_words(
                node,
                av_letters,
                best_word.clone(),
                y,
                is_vertical,
                word,
                can_be,
                points,
                x + 1,
            )
        }

        best_word
    }

    fn place_best_word(&mut self) -> Word {
        let mut best_word = Word::new();

        for y in 0..15 {
            if self.game.board[y]
                .iter()
                .filter(|&letter| *letter == '-')
                .count()
                == 15
                && (y == 0
                    || self.game.board[y - 1]
                        .iter()
                        .filter(|&letter| *letter == '-')
                        .count()
                        == 15)
                && (y == 14
                    || self.game.board[y + 1]
                        .iter()
                        .filter(|&letter| *letter == '-')
                        .count()
                        == 15)
            {
                continue;
            }

            best_word = self.find_words(
                &NODE,
                &self.letters,
                best_word,
                y,
                false,
                "".to_string(),
                [false, false],
                [0, 0, 1],
                0,
            );
        }

        self.game.board = (0..15)
            .map(|col| (0..15).map(|row| self.game.board[row][col]).collect())
            .collect();

        for y in 0..15 {
            if self.game.board[y]
                .iter()
                .filter(|&letter| *letter == '-')
                .count()
                == 15
                && (y == 0
                    || self.game.board[y - 1]
                        .iter()
                        .filter(|&letter| *letter == '-')
                        .count()
                        == 15)
                && (y == 14
                    || self.game.board[y + 1]
                        .iter()
                        .filter(|&letter| *letter == '-')
                        .count()
                        == 15)
            {
                continue;
            }

            best_word = self.find_words(
                &NODE,
                &self.letters,
                best_word,
                y,
                true,
                "".to_string(),
                [false, false],
                [0, 0, 1],
                0,
            );
        }

        self.game.board = (0..15)
            .map(|col| (0..15).map(|row| self.game.board[row][col]).collect())
            .collect();

        if best_word.score != 0 {
            self.game.insert_word(&mut best_word);
            self.letters = best_word.av_letters.clone();
            self.score += best_word.score;
            self.get_new_letters();
        }

        best_word
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

pub fn play_game(n: u16) {
    let mut game: Game = Game::new();
    let mut player1: Player = Player::new(&mut game);

    println!("{:?}", player1.letters);
    println!("{:?}", player1.make_move(true));

    for _ in 0..n {
        println!("{:?}", player1.letters);
        println!("{:?}", player1.make_move(false));
    }
    println!("{}", game);
}

#[derive(Serialize, Deserialize)]
struct Person {
    name: String,
    age: u8,
    phones: Vec<String>,
}

fn main() {
    play_game(15);
}
