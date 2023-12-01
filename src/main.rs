#![allow(unused, non_snake_case)]
use itertools::Itertools;
use my_lib::*;
use procon_input::*;
use rand::prelude::*;
use rand_pcg::Mcg128Xsl64;
use std::{
    cell::RefCell,
    clone,
    cmp::{max, min},
    collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque},
    iter::FromIterator,
    mem::swap,
    ops::*,
    rc::Rc,
    slice::SliceIndex,
};

use crate::solver::FirstCleaning;

fn main() {
    let start_time = my_lib::time::update();

    Sim::new().run();

    let end_time = my_lib::time::update();
    let duration = end_time - start_time;
    eprintln!("{:?} ", duration);
}

#[derive(Debug, Clone)]
pub struct State {
    score: usize,
}

impl State {
    fn new() -> Self {
        State { score: 0 }
    }

    fn change(&mut self, output: &mut Output, rng: &mut Mcg128Xsl64) {
        //let val = rng.gen_range(-3, 4);
        //self.x += val;
    }

    fn compute_score(&mut self) {
        //self.score = 0;
    }
}

#[derive(Debug, Clone)]
pub struct Sim {
    input: Input,
}

impl Sim {
    fn new() -> Self {
        // TODO: impl input
        let input = Input::read();
        Sim { input }
    }

    pub fn run(&mut self) {
        let mut output = Output::new();
        let mut first_cleaning = FirstCleaning::new(self.input.clone());
        first_cleaning.run(&mut output);
        output.submit();
        let (score, _) = Tools::compute_score(&self.input, &output);

        let mut rng: Mcg128Xsl64 = rand_pcg::Pcg64Mcg::new(890482);
        let mut cnt = 0 as usize; // 試行回数

        // //let mut initial_state = State::new();
        // let mut best_output = Output::new();
        // let mut best_state = State::new();
        // best_state.compute_score();

        // 'outer: loop {
        //     let current_time = my_lib::time::update();
        //     if current_time >= my_lib::time::LIMIT {
        //         break;
        //     }

        //     cnt += 1;

        //     let mut output = Output::new();

        //     // A:近傍探索
        //     let mut state: State = best_state.clone();
        //     state.change(&mut output, &mut rng);

        //     // B:壊して再構築
        //     // best_outputの一部を破壊して、それまでのoutputを使ってstateを作り直して再構築したり
        //     // outputの変形
        //     // best_output.remove(&mut output, &mut rng);
        //     // let mut state: State = initial_state.clone();
        //     // stateを新outputの情報で復元
        //     // そこから続きやる

        //     // スコア計算
        //     state.compute_score();

        //     // 状態更新
        //     solver::mountain(&mut best_state, &state, &mut best_output, &output);
        //     //solver::simulated_annealing(&mut best_state, &state, &mut best_output, &output, self.current_time, &mut rng);
        // }

        //best_output.submit();

        eprintln!("{} ", cnt);
        let best_score = score;
        eprintln!("{} ", best_score);
    }
}

#[derive(Clone, Debug)]
pub struct Input {
    pub N: usize,
    pub h: Vec<Vec<char>>,
    pub v: Vec<Vec<char>>,
    pub d: Vec<Vec<i64>>,
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.N)?;
        for i in 0..self.N - 1 {
            writeln!(f, "{}", self.h[i].iter().collect::<String>())?;
        }
        for i in 0..self.N {
            writeln!(f, "{}", self.v[i].iter().collect::<String>())?;
        }
        for i in 0..self.N {
            writeln!(f, "{}", self.d[i].iter().join(" "))?;
        }
        Ok(())
    }
}

impl Input {
    fn read() -> Self {
        let N = read_u();

        let mut h: Vec<Vec<char>> = vec![vec!['0'; N]; N - 1];
        for i in 0..N - 1 {
            let c_vec = read_line_as_chars();
            for j in 0..N {
                h[i][j] = c_vec[j];
            }
        }

        let mut v: Vec<Vec<char>> = vec![vec!['0'; N - 1]; N];
        for i in 0..N {
            let c_vec = read_line_as_chars();
            for j in 0..N - 1 {
                v[i][j] = c_vec[j];
            }
        }

        let mut d: Vec<Vec<i64>> = vec![vec![0; N]; N];
        for i in 0..N {
            let mut d_vec = read_i_vec();
            for j in 0..N {
                d[i][j] = d_vec[j];
            }
        }

        Input { N, h, v, d }
    }
}

#[derive(Debug, Clone)]
pub struct Output {
    out: Vec<char>,
}

impl Output {
    fn new() -> Self {
        Output { out: vec![] }
    }

    fn remove(&self, output: &mut Self, rng: &mut Mcg128Xsl64) {
        // https://atcoder.jp/contests/ahc014/submissions/35567589 L558
    }

    fn submit(&self) {
        println!("{}", self.out.iter().collect::<String>());
    }
}

mod solver {
    use super::*;

    pub struct FirstCleaning {
        input: Input,
    }

    impl FirstCleaning {
        pub fn new(input: Input) -> Self {
            FirstCleaning { input }
        }

        pub fn run(self, output: &mut Output) {
            let mut has_seen = vec![vec![false; self.input.N]; self.input.N];

            let start = (0, 0);
            self.dfs(start, &mut has_seen, output);
        }

        fn dfs(&self, (i, j): (usize, usize), has_seen: &mut Vec<Vec<bool>>, output: &mut Output) {
            has_seen[i][j] = true;
            for dir in 0..4_usize {
                let (di, dj) = DIJ[dir];
                let (ni, nj) = (i + di, j + dj);
                if ni >= self.input.N || nj >= self.input.N {
                    continue;
                }
                if has_seen[ni][nj] {
                    continue;
                }
                if (di == 0 && self.input.v[i][min(j, nj)] == '0')
                    || (dj == 0 && self.input.h[min(i, ni)][j] == '0')
                {
                    let c = DIR.chars().nth(dir).unwrap();
                    output.out.push(c);
                    self.dfs((ni, nj), has_seen, output);
                    let c = DIR.chars().nth((dir + 2) % 4).unwrap();
                    output.out.push(c);
                }
            }
        }
    }

    pub fn mountain(
        best_state: &mut State,
        state: &State,
        best_output: &mut Output,
        output: &Output,
    ) {
        //! bese_state(self)を更新する。

        // 最小化の場合は > , 最大化の場合は < 。
        if best_state.score > state.score {
            *best_state = state.clone();
            *best_output = output.clone();
        }
    }

    const T0: f64 = 2e3;
    //const T1: f64 = 6e2; // 終端温度が高いと最後まで悪いスコアを許容する
    const T1: f64 = 6e1; // 終端温度が高いと最後まで悪いスコアを許容する
    pub fn simulated_annealing(
        best_state: &mut State,
        state: &State,
        best_output: &mut Output,
        output: &Output,
        current_time: f64,
        rng: &mut Mcg128Xsl64,
    ) {
        //! 焼きなまし法
        //! https://scrapbox.io/minyorupgc/%E7%84%BC%E3%81%8D%E3%81%AA%E3%81%BE%E3%81%97%E6%B3%95

        static mut T: f64 = T0;
        static mut CNT: usize = 0;
        let temperature = unsafe {
            CNT += 1;
            if CNT % 100 == 0 {
                let t = current_time / my_lib::time::LIMIT;
                T = T0.powf(1.0 - t) * T1.powf(t);
            }
            T
        };

        // 最大化の場合
        let delta = (best_state.score as f64) - (state.score as f64);
        // 最小化の場合
        //let delta = (state.score as f64) - (best_state.score as f64);

        let prob = f64::exp(-delta / temperature).min(1.0);

        if delta < 0.0 {
            *best_state = state.clone();
            *best_output = output.clone();
        } else if rng.gen_bool(prob) {
            *best_state = state.clone();
            *best_output = output.clone();
        }
    }
}

mod my_lib {
    //! 基本的に問題によらず変えない自作ライブラリ群
    use super::*;
    pub mod time {
        //! 時間管理モジュール
        pub fn update() -> f64 {
            static mut STARTING_TIME_MS: Option<f64> = None;
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            let time_ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
            unsafe {
                let now = match STARTING_TIME_MS {
                    Some(starting_time_ms) => time_ms - starting_time_ms,
                    None => {
                        STARTING_TIME_MS = Some(time_ms);
                        0.0 as f64
                    }
                };
                now
            }
        }

        // TODO: set LIMIT
        pub const LIMIT: f64 = 0.3;
    }

    pub trait Mat<S, T> {
        fn set(&mut self, p: S, value: T);
        fn get(&self, p: S) -> T;
        fn swap(&mut self, p1: S, p2: S);
    }

    impl<T> Mat<&Point, T> for Vec<Vec<T>>
    where
        T: Copy,
    {
        fn set(&mut self, p: &Point, value: T) {
            self[p.y][p.x] = value;
        }

        fn get(&self, p: &Point) -> T {
            self[p.y][p.x]
        }

        fn swap(&mut self, p1: &Point, p2: &Point) {
            let tmp = self[p1.y][p1.x];
            self[p1.y][p1.x] = self[p2.y][p2.x];
            self[p2.y][p2.x] = tmp;
        }
    }

    impl<T> Mat<Point, T> for Vec<Vec<T>>
    where
        T: Copy,
    {
        fn set(&mut self, p: Point, value: T) {
            self[p.y][p.x] = value;
        }

        fn get(&self, p: Point) -> T {
            self[p.y][p.x]
        }

        fn swap(&mut self, p1: Point, p2: Point) {
            let tmp = self[p1.y][p1.x];
            self[p1.y][p1.x] = self[p2.y][p2.x];
            self[p2.y][p2.x] = tmp;
        }
    }

    impl Add for Point {
        type Output = Result<Point, &'static str>;
        fn add(self, rhs: Self) -> Self::Output {
            let (x, y) = if cfg!(debug_assertions) {
                // debugではオーバーフローでpanic発生するため、オーバーフローの溢れを明確に無視する(※1.60場合。それ以外は不明)
                (self.x.wrapping_add(rhs.x), self.y.wrapping_add(rhs.y))
            } else {
                (self.x + rhs.x, self.y + rhs.y)
            };

            unsafe {
                if let Some(width) = WIDTH {
                    if x >= width || y >= width {
                        return Err("out of range");
                    }
                }
            }

            Ok(Point { x, y })
        }
    }

    static mut WIDTH: Option<usize> = None;

    #[derive(Debug, Clone, PartialEq, Eq, Copy)]
    pub struct Point {
        pub x: usize, // →
        pub y: usize, // ↑
    }

    impl Point {
        pub fn new(x: usize, y: usize) -> Self {
            Point { x, y }
        }

        pub fn set_width(width: usize) {
            unsafe {
                WIDTH = Some(width);
            }
        }
    }

    pub trait SortFloat {
        fn sort(&mut self);
        fn sort_rev(&mut self);
    }

    impl SortFloat for Vec<f64> {
        fn sort(&mut self) {
            //! 浮動小数点としてNANが含まれないことを約束されている場合のsort処理<br>
            //! 小さい順
            self.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        fn sort_rev(&mut self) {
            //! 浮動小数点としてNANが含まれないことを約束されている場合のsort処理<br>  
            //! 大きい順
            self.sort_by(|a, b| b.partial_cmp(a).unwrap());
        }
    }

    pub trait EvenOdd {
        fn is_even(&self) -> bool;
        fn is_odd(&self) -> bool;
    }

    impl EvenOdd for usize {
        fn is_even(&self) -> bool {
            self % 2 == 0
        }

        fn is_odd(&self) -> bool {
            self % 2 != 0
        }
    }
}

mod procon_input {
    use std::{any::type_name, io::*};

    fn read_block<T: std::str::FromStr>() -> T {
        let mut s = String::new();
        let mut buf = [0];
        loop {
            stdin().read(&mut buf).expect("can't read.");
            let c = buf[0] as char;
            if c == ' ' {
                break;
            }
            // for Linux
            if c == '\n' {
                break;
            }
            // for Windows
            if c == '\r' {
                // pop LR(line feed)
                stdin().read(&mut buf).expect("can't read.");
                break;
            }
            s.push(c);
        }
        s.parse::<T>()
            .unwrap_or_else(|_| panic!("can't parse '{}' to {}", s, type_name::<T>()))
    }

    pub fn read_i() -> i64 {
        read_block::<i64>()
    }

    pub fn read_ii() -> (i64, i64) {
        (read_block::<i64>(), read_block::<i64>())
    }

    pub fn read_iii() -> (i64, i64, i64) {
        (
            read_block::<i64>(),
            read_block::<i64>(),
            read_block::<i64>(),
        )
    }

    pub fn read_iiii() -> (i64, i64, i64, i64) {
        (
            read_block::<i64>(),
            read_block::<i64>(),
            read_block::<i64>(),
            read_block::<i64>(),
        )
    }

    pub fn read_u() -> usize {
        read_block::<usize>()
    }

    pub fn read_uu() -> (usize, usize) {
        (read_block::<usize>(), read_block::<usize>())
    }

    pub fn read_uuu() -> (usize, usize, usize) {
        (
            read_block::<usize>(),
            read_block::<usize>(),
            read_block::<usize>(),
        )
    }

    pub fn read_uuuu() -> (usize, usize, usize, usize) {
        (
            read_block::<usize>(),
            read_block::<usize>(),
            read_block::<usize>(),
            read_block::<usize>(),
        )
    }

    pub fn read_f() -> f64 {
        read_block::<f64>()
    }

    pub fn read_ff() -> (f64, f64) {
        (read_block::<f64>(), read_block::<f64>())
    }

    pub fn read_c() -> char {
        read_block::<char>()
    }

    pub fn read_cc() -> (char, char) {
        (read_block::<char>(), read_block::<char>())
    }

    fn read_line() -> String {
        let mut s = String::new();
        stdin().read_line(&mut s).expect("can't read.");
        s.trim()
            .parse()
            .unwrap_or_else(|_| panic!("can't trim in read_line()"))
    }

    pub fn read_vec<T: std::str::FromStr>() -> Vec<T> {
        read_line()
            .split_whitespace()
            .map(|e| {
                e.parse()
                    .unwrap_or_else(|_| panic!("can't parse '{}' to {}", e, type_name::<T>()))
            })
            .collect()
    }

    pub fn read_i_vec() -> Vec<i64> {
        read_line()
            .split_whitespace()
            .map(|e| {
                e.parse()
                    .unwrap_or_else(|_| panic!("can't parse '{}' to {}", e, type_name::<i64>()))
            })
            .collect()
    }

    pub fn read_u_vec() -> Vec<usize> {
        read_line()
            .split_whitespace()
            .map(|e| {
                e.parse()
                    .unwrap_or_else(|_| panic!("can't parse '{}' to {}", e, type_name::<usize>()))
            })
            .collect()
    }

    pub fn read_f_vec() -> Vec<f64> {
        read_line()
            .split_whitespace()
            .map(|e| {
                e.parse()
                    .unwrap_or_else(|_| panic!("can't parse '{}' to {}", e, type_name::<f64>()))
            })
            .collect()
    }

    pub fn read_c_vec() -> Vec<char> {
        read_line()
            .split_whitespace()
            .map(|e| {
                e.parse()
                    .unwrap_or_else(|_| panic!("can't parse '{}' to {}", e, type_name::<char>()))
            })
            .collect()
    }

    pub fn read_line_as_chars() -> Vec<char> {
        //! a b c d -> \[a, b, c, d]
        read_line().as_bytes().iter().map(|&b| b as char).collect()
    }

    pub fn read_string() -> String {
        //! abcd -> "abcd"
        read_block::<String>()
    }

    pub fn read_string_as_chars() -> Vec<char> {
        //! abcd -> \[a, b, c, d]
        read_block::<String>().chars().collect::<Vec<char>>()
    }
}

const DIR: &str = "RDLU";
const DIJ: [(usize, usize); 4] = [(0, 1), (1, 0), (0, !0), (!0, 0)];

mod Tools {
    use super::*;

    #[macro_export]
    macro_rules! mat {
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

    #[derive(Clone, Debug)]
    pub struct Eval {
        pub score: i64,
        pub err: String,
        pub d: Vec<Vec<i64>>,
        pub route: Vec<(usize, usize)>,
        pub last_visited: Vec<Vec<usize>>,
        pub edge_count: Vec<Vec<(i32, i32)>>,
        pub S: Vec<i64>,
        pub average: Vec<Vec<f64>>,
    }

    impl Eval {
        fn get_a(&self, t: usize) -> Vec<Vec<i64>> {
            let N = self.d.len();
            let L = self.route.len() - 1;
            let mut a = mat![0; N; N];
            let mut last_visited2 = self.last_visited.clone();
            for t in L..L + t {
                let (i, j) = self.route[t - L];
                last_visited2[i][j] = t;
            }
            for i in 0..N {
                for j in 0..N {
                    a[i][j] = (L + t - last_visited2[i][j]) as i64 * self.d[i][j];
                }
            }
            let (i, j) = self.route[t];
            a[i][j] = 0;
            a
        }
    }

    fn can_move(
        N: usize,
        h: &Vec<Vec<char>>,
        v: &Vec<Vec<char>>,
        i: usize,
        j: usize,
        dir: usize,
    ) -> bool {
        let (di, dj) = DIJ[dir];
        let i2 = i + di;
        let j2 = j + dj;
        if i2 >= N || j2 >= N {
            return false;
        }
        if di == 0 {
            v[i][j.min(j2)] == '0'
        } else {
            h[i.min(i2)][j] == '0'
        }
    }

    fn evaluate(input: &Input, out: &[char]) -> Eval {
        let mut last_visited = mat![!0; input.N; input.N];
        let L = out.len();
        let mut i = 0;
        let mut j = 0;
        let mut route = vec![];
        let mut S = vec![];
        let mut average = mat![0.0; input.N; input.N];
        let mut edge_count = mat![(0, 0); input.N; input.N];
        for t in 0..L {
            route.push((i, j));
            last_visited[i][j] = t;
            if let Some(dir) = DIR.find(out[t]) {
                if can_move(input.N, &input.h, &input.v, i, j, dir) {
                    if DIJ[dir].0 == 0 {
                        edge_count[i][j.min(j + DIJ[dir].1)].0 += 1;
                    } else {
                        edge_count[i.min(i + DIJ[dir].0)][j].1 += 1;
                    }
                    i += DIJ[dir].0;
                    j += DIJ[dir].1;
                } else {
                    return Eval {
                        score: 0,
                        err: format!("The output route hits a wall."),
                        d: input.d.clone(),
                        route,
                        last_visited: mat![0; input.N; input.N],
                        S,
                        average,
                        edge_count,
                    };
                }
            } else {
                return Eval {
                    score: 0,
                    err: format!("Illegal output char: {}", out[t]),
                    d: input.d.clone(),
                    route,
                    last_visited: mat![0; input.N; input.N],
                    S,
                    average,
                    edge_count,
                };
            }
        }
        route.push((i, j));
        if (i, j) != (0, 0) {
            return Eval {
                score: 0,
                err: format!("The output route does not return to (0, 0)."),
                d: input.d.clone(),
                route,
                last_visited: mat![0; input.N; input.N],
                S,
                average,
                edge_count,
            };
        }
        for i in 0..input.N {
            for j in 0..input.N {
                if last_visited[i][j] == !0 {
                    return Eval {
                        score: 0,
                        err: format!("The output route does not visit ({}, {}).", i, j),
                        d: input.d.clone(),
                        route,
                        last_visited: mat![0; input.N; input.N],
                        S,
                        average,
                        edge_count,
                    };
                }
            }
        }
        let mut s = 0;
        let mut sum_d = 0;
        for i in 0..input.N {
            for j in 0..input.N {
                s += (L - last_visited[i][j]) as i64 * input.d[i][j];
                sum_d += input.d[i][j];
            }
        }
        let mut last_visited2 = last_visited.clone();
        let mut sum = mat![0; input.N; input.N];
        for t in L..2 * L {
            let (i, j) = route[t - L];
            let dt = (t - last_visited2[i][j]) as i64;
            let a = dt * input.d[i][j];
            sum[i][j] += dt * (dt - 1) / 2 * input.d[i][j];
            s -= a;
            last_visited2[i][j] = t;
            S.push(s);
            s += sum_d;
        }
        for i in 0..input.N {
            for j in 0..input.N {
                average[i][j] = sum[i][j] as f64 / L as f64;
            }
        }
        let score = (2 * S.iter().sum::<i64>() + L as i64) / (2 * L) as i64;
        Eval {
            score,
            err: String::new(),
            d: input.d.clone(),
            route,
            last_visited,
            S,
            average,
            edge_count,
        }
    }

    pub fn compute_score(input: &Input, out: &Output) -> (i64, String) {
        let ret = evaluate(input, &out.out);
        if ret.err.len() > 0 {
            (0, ret.err)
        } else {
            (ret.score, ret.err)
        }
    }
}
