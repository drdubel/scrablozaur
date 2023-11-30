use std::collections::HashMap;
use std::fmt::{Display, Formatter, Result};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead};
use std::path::Path;

#[derive(Debug, Clone, Eq)]
struct Node {
    children: HashMap<char, Node>,
    is_terminal: bool,
    id: u32,
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
        Node {
            children: HashMap::new(),
            is_terminal: false,
            id: Node::get_next_id(),
        }
    }

    fn __repr__(&self) -> String {
        format!("{}", self)
    }

    fn get_next_id() -> u32 {
        static mut NEXT_ID: u32 = 0;

        unsafe {
            let id = NEXT_ID;
            NEXT_ID += 1;
            id
        }
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

fn read_lines<P>(filename: P) -> io::Result<Vec<String>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file)
        .lines()
        .filter_map(|line| line.ok())
        .collect())
}

fn get_pref_len(prev_word: String, word: String) -> u8 {
    let mut pref_len: u8 = 0;

    for (char1, char2) in prev_word.chars().zip(word.chars()) {
        if char1 != char2 {
            return pref_len;
        }

        pref_len += 1;
    }
    pref_len
}

fn minimize<'a>(
    mut curr_node: &mut Node,
    pref_len: u8,
    mut minimized_nodes: HashMap<Node, Node>,
    mut non_minimized_nodes: Vec<(&mut Node, char, &mut Node)>,
) -> (
    &'a mut Node,
    HashMap<Node, Node>,
    Vec<(&'a mut Node, char, &'a mut Node)>,
) {
    for _ in 0..(non_minimized_nodes.len() as u8 - pref_len) {
        let (mut parent, letter, child) = non_minimized_nodes.pop().unwrap();

        if !minimized_nodes.contains_key(&child) {
            minimized_nodes.entry(child.clone()).or_insert(*child);
        } else {
            parent.children.entry(letter).or_insert(*child);
        }

        curr_node = parent;
    }

    (curr_node, minimized_nodes, non_minimized_nodes)
}

fn build_dawg(words: Vec<String>) -> Node {
    let mut root = Node::new();
    let mut minimized_nodes: HashMap<Node, Node> = HashMap::from([(root.clone(), root.clone())]);
    let mut non_minimized_nodes: Vec<(&mut Node, char, &mut Node)> = Vec::new();
    let mut curr_node: &mut Node = &mut root;
    let mut prev_word: String = "".to_string();

    for word in words {
        let pref_len = get_pref_len(prev_word, word.to_string());

        if !non_minimized_nodes.is_empty() {
            (curr_node, minimized_nodes, non_minimized_nodes) =
                minimize(curr_node, pref_len, minimized_nodes, non_minimized_nodes);
        }

        for letter in word[(pref_len as usize)..].chars() {
            let mut next_node = Node::new();
            curr_node.children.entry(letter).or_insert(next_node);
            non_minimized_nodes.push((curr_node, letter, &mut next_node));
            curr_node = &mut next_node;
        }

        curr_node.is_terminal = true;
        prev_word = word.to_string();
    }

    minimize(curr_node, 0, minimized_nodes.clone(), non_minimized_nodes);
    println!("{}", minimized_nodes.len());

    root
}

fn main() {
    if let Ok(words) = read_lines("src/words.txt") {
        let dawg = build_dawg(words);
        println!("{:?}", dawg.children);
    } else {
        println!("NIE");
    }
}
