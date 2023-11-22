use pyo3::prelude::*;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result};
use std::fs;
use std::hash::{Hash, Hasher};

static mut NEXT_ID: i32 = 0;

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
            tile_bag: vec![
                'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'ą', 'b', 'b', 'c', 'c', 'c', 'ć',
                'd', 'd', 'd', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'ę', 'f', 'g', 'g', 'h', 'h',
                'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'j', 'j', 'k', 'k', 'k', 'l', 'l', 'l',
                'ł', 'ł', 'm', 'm', 'm', 'n', 'n', 'n', 'n', 'n', 'ń', 'o', 'o', 'o', 'o', 'o',
                'o', 'ó', 'p', 'p', 'p', 'r', 'r', 'r', 'r', 's', 's', 's', 's', 'ś', 't', 't',
                't', 'u', 'u', 'w', 'w', 'w', 'w', 'y', 'y', 'y', 'y', 'z', 'z', 'z', 'z', 'z',
                'ź', 'ż',
            ],
        }
    }

    fn __repr__(&self) -> String {
        format!("{}", self)
    }

    fn insert_word(&mut self, orientation: bool, mut pos: Vec<i16>, word: String) {
        if orientation {
            self.board = (0..15)
                .map(|col| (0..15).map(|row| self.board[row][col]).collect())
                .collect();
            pos.reverse();
        }
        for i in pos[1]..(pos[1] + word.len() as i16) {
            self.board[pos[0] as usize][i as usize] =
                word.chars().nth((i - pos[1]) as usize).unwrap();
        }
        if orientation {
            self.board = (0..15)
                .map(|col| (0..15).map(|row| self.board[row][col]).collect())
                .collect();
        }
    }

    fn give_new_letters(&mut self, letters: &Vec<char>) -> Vec<char> {
        let mut rng = rand::thread_rng();
        let new_letters: Vec<char> = (0..(7 - letters.len()).min(self.tile_bag.len()))
            .filter_map(|_| letters.choose(&mut rng))
            .cloned()
            .collect();
        for letter in new_letters.iter() {
            let index = self.tile_bag.iter().position(|x| *x == *letter).unwrap();
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
            letters: Vec::new(),
            score: 0,
            game,
        }
    }

    fn exchange_letters(&mut self, n: i16) {
        let mut rng = rand::thread_rng();
        self.letters = (0..(7 - n).min(self.letters.len() as i16))
            .filter_map(|_| self.letters.choose(&mut rng))
            .cloned()
            .collect();
        self.get_new_letters();
    }

    fn get_new_letters(&mut self) {
        let new_letters = self.game.give_new_letters(&self.letters);
        self.letters.extend(new_letters);
    }

    fn validate_word(&self, node: Node, word: String, x: i16) -> String {
        if x == word.len() as i16 {
            if node.is_terminal {
                return word;
            }
            return "".to_string();
        }
        if node
            .children
            .contains_key(&word.chars().nth(x as usize).unwrap())
        {
            return self.validate_word(
                node.children.get(&word.chars().nth(x as usize).unwrap()),
                word,
                x + 1,
            );
        }
        return "".to_string();
    }
}

#[pyfunction]
fn play_game() {
    let data = fs::read_to_string("./dawg.json").expect("Unable to read file");
    let node: Node = serde_json::from_str(&data).expect("JSON does not have correct format.");
    println!("{}", node);
    let mut game: Game = Game::new(node);
    let player1: Player = Player::new(&mut game);
    println!("{}", game);
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
