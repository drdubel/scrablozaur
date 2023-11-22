use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result};
use std::fs;
use std::hash::{Hash, Hasher};

static mut NEXT_ID: i32 = 0;

#[derive(Serialize, Deserialize, Debug, Clone, Eq)]
#[pyclass(get_all, set_all, subclass)]
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

#[pymethods]
impl Node {
    #[new]
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

#[derive(Debug)]
#[pyclass(get_all, set_all, subclass)]
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

#[pymethods]
impl Game {
    #[new]
    fn new(dawg: Node) -> Self {
        Game {
            dawg,
            board: (0..15).map(|_| (0..15).map(|_| '-').collect()).collect(),
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
}

#[pyclass(get_all, set_all, subclass)]
struct Player {
    points: i16,
    letters: Vec<char>,
    best_word: String,
}

#[pymethods]
impl Player {
    #[new]
    fn new() -> Self {
        Player {
            points: 0,
            letters: Vec::new(),
            best_word: "".to_string(),
        }
    }
}

#[pyfunction]
fn play_game() {
    let data = fs::read_to_string("./dawg.json").expect("Unable to read file");
    let node: Node = serde_json::from_str(&data).expect("JSON does not have correct format.");
    println!("{}", node);
    let game: Game = Game::new(node);
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
    m.add_class::<Node>()?;
    m.add_class::<Player>()?;
    m.add_class::<Game>()?;
    m.add_function(wrap_pyfunction!(play_game, m)?)?;
    Ok(())
}
