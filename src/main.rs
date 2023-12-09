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

use crate::Tools::compute_score;

fn main() {
    let start_time = my_lib::time::update();

    Sim::new().run();

    let end_time = my_lib::time::update();
    let duration = end_time - start_time;
    eprintln!("{:?} ", duration);
}

type DistMap = Vec<Vec<usize>>;

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
        let mut rng: Mcg128Xsl64 = rand_pcg::Pcg64Mcg::new(890482);

        let mut cnt = 0 as usize; // 試行回数

        let dist_maps = solver::create_dist_maps(&self.input);

        let mut acts_map = BTreeMap::new();

        let entry_pos = (0, 0);
        let mut solve = solver::CleanAroundHighA::new(self.input.N);
        let mut current_day = 0;
        let max_clean_cnt = std::usize::MAX;
        let mut output = Output::new();
        solve.run(
            &self.input,
            entry_pos,
            &mut current_day,
            &mut output,
            max_clean_cnt,
            &mut acts_map,
            &dist_maps,
        );
        let (score, _) = Tools::compute_score(&self.input, &output);
        //areas.debug_id_map();

        output.submit();

        eprintln!("{} ", cnt);
        let score = if output.out.len() >= 100000 {
            -1
        } else {
            score
        };
        eprintln!("{} ", score);
        eprintln!("{} ", self.input.N);
        eprintln!("{} ", output.out.len());
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

    fn add(&mut self, actions: &Vec<char>) {
        for &act in actions.iter() {
            self.out.push(act);
        }
    }

    fn submit(&self) {
        println!("{}", self.out.iter().collect::<String>());
    }
}

mod solver {
    use std::vec;

    use super::*;

    #[derive(Clone, Debug)]
    pub struct CleanAroundHighA {}

    impl CleanAroundHighA {
        pub fn new(N: usize) -> Self {
            CleanAroundHighA {}
        }

        pub fn run(
            &mut self,
            input: &Input,
            mut entry_pos: (usize, usize),
            current_day: &mut usize,
            output: &mut Output,
            max_clean_cnt: usize,
            acts_map: &mut BTreeMap<((usize, usize), (usize, usize)), Vec<char>>,
            dist_maps: &Vec<Vec<DistMap>>,
        ) {
            let first_pos = solver::decide_start_point(&input);

            let mut remains = BTreeSet::new();
            for i in 0..input.N {
                for j in 0..input.N {
                    remains.insert((i, j));
                }
            }

            let mut prev_day = vec![vec![0; input.N]; input.N];

            let from = entry_pos;
            let to = first_pos;
            move_between_two_points(
                input,
                from,
                to,
                output,
                acts_map,
                current_day,
                &mut prev_day,
                &mut remains,
                dist_maps,
            );

            let mut current_pos: (usize, usize) = to;
            let start_time = my_lib::time::update();
            for cnt in 0..max_clean_cnt {
                if output.out.len() >= 98000 {
                    break;
                }
                let current_time = my_lib::time::update();
                if current_time - start_time >= my_lib::time::LIMIT {
                    break;
                }

                clean_large_a(
                    input,
                    current_day,
                    &mut current_pos,
                    acts_map,
                    output,
                    &mut prev_day,
                    &mut remains,
                    dist_maps,
                );

                if cnt != 0 && cnt % 3 == 0 {
                    let len = remains.len() / 3;
                    let mut pos_set = BTreeSet::new();
                    for &(i, j) in remains.iter().take(len) {
                        let di = (i as i64 - current_pos.0 as i64).abs();
                        let dj = (j as i64 - current_pos.1 as i64).abs();
                        let dist = (di + dj) as usize;
                        if dist > input.N {
                            continue;
                        }
                        pos_set.insert((i, j));
                    }
                    move_around_pos_set(
                        input,
                        current_day,
                        &mut current_pos,
                        acts_map,
                        &mut pos_set,
                        output,
                        &mut prev_day,
                        &mut remains,
                        dist_maps,
                    );
                }
            }

            move_around_pos_set(
                input,
                current_day,
                &mut current_pos,
                acts_map,
                &mut remains,
                output,
                &mut prev_day,
                &mut BTreeSet::new(),
                dist_maps,
            );

            let from = current_pos;
            let to = (0, 0);
            move_between_two_points(
                input,
                from,
                to,
                output,
                acts_map,
                current_day,
                &mut prev_day,
                &mut remains,
                dist_maps,
            );
        }
    }

    pub fn clean_large_a(
        input: &Input,
        current_day: &mut usize,
        last_pos: &mut (usize, usize),
        acts_map: &mut BTreeMap<((usize, usize), (usize, usize)), Vec<char>>,
        output: &mut Output,
        prev_day: &mut Vec<Vec<usize>>,
        remains: &mut BTreeSet<(usize, usize)>,
        dist_maps: &Vec<Vec<DistMap>>,
    ) {
        let entry_pos = *last_pos;

        let dist_map = &dist_maps[entry_pos.0][entry_pos.1];
        // aの高い順にソート
        let mut a_ranking = vec![];
        for i in 0..input.N {
            for j in 0..input.N {
                let dist = dist_map[i][j];
                let diff = (*current_day - prev_day[i][j] + 1) as i64;
                let d = input.d[i][j];
                let a = d * diff;
                a_ranking.push((a, (i, j)));
            }
        }

        // 降順ソート
        a_ranking.sort_by(|a, b| b.0.cmp(&a.0));

        let high_a = a_ranking[0].0;

        let num_devide = 5;

        let cnt_max = a_ranking.len() / num_devide;
        let mut cnt = 0;
        let mut pos_set = BTreeSet::new();

        for &(a, (i, j)) in a_ranking.iter() {
            cnt += 1;
            pos_set.insert((i, j));

            if input.d[i][j] == 0 {
                break;
            }

            if input.d[i][j] < 50 && a < high_a / 4 && cnt >= cnt_max {
                break;
            }
        }

        // entry_posから近い点に移動。その点からさらに近い点に移動。を繰り返す
        move_around_pos_set(
            input,
            current_day,
            last_pos,
            acts_map,
            &mut pos_set,
            output,
            prev_day,
            remains,
            dist_maps,
        );
    }

    pub fn remove_remains_from_actions(
        entry_pos: (usize, usize),
        actions: &Vec<char>,
        remains: &mut BTreeSet<(usize, usize)>,
    ) {
        let mut pos = entry_pos;
        for act in actions.iter() {
            match act {
                'R' => pos.1 += 1,
                'L' => pos.1 -= 1,
                'D' => pos.0 += 1,
                'U' => pos.0 -= 1,
                _ => unreachable!(),
            }
            remains.remove(&pos);
        }
    }

    pub fn decide_start_point(input: &Input) -> (usize, usize) {
        let mut start = (0, 0);
        let mut voted_d_map = vec![vec![0; input.N]; input.N];
        let mut voted_cnt_map = vec![vec![0; input.N]; input.N];
        for i in 0..input.N {
            for j in 0..input.N {
                voted_d_map[i][j] = input.d[i][j];
                voted_cnt_map[i][j] += 1;

                let range = [!0, 0, 1];
                for &di in range.iter() {
                    for &dj in range.iter() {
                        let (ni, nj) = (i + di, j + dj);
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        if ni >= input.N || nj >= input.N {
                            continue;
                        }
                        if (di == 0 && input.v[i][min(j, nj)] == '0')
                            || (dj == 0 && input.h[min(i, ni)][j] == '0')
                        {
                            let dist = (i as i64 - ni as i64).abs() + (j as i64 - nj as i64).abs();
                            voted_d_map[i][j] += input.d[ni][nj] / (dist + 1);
                            voted_cnt_map[i][j] += 1;
                        }
                    }
                }
            }
        }

        let mut max_d = 0;
        let mut averaged_d_map = vec![vec![0; input.N]; input.N];
        for i in 0..input.N {
            for j in 0..input.N {
                if voted_cnt_map[i][j] == 0 {
                    continue;
                }
                averaged_d_map[i][j] = voted_d_map[i][j] / voted_cnt_map[i][j];
                let d = voted_d_map[i][j] / voted_cnt_map[i][j];
                if d > max_d {
                    max_d = d;
                    start = (i, j);
                }
            }
        }
        start
    }

    pub fn create_dist_maps(input: &Input) -> Vec<Vec<DistMap>> {
        let mut dist_maps: Vec<Vec<DistMap>> =
            vec![vec![vec![vec![std::usize::MAX; input.N]; input.N]; input.N]; input.N];
        for i in 0..input.N {
            for j in 0..input.N {
                let dist_map = compute_dist_to_all_pos(input, (i, j));
                dist_maps[i][j] = dist_map;
            }
        }
        dist_maps
    }

    pub fn compute_dist_to_all_pos(input: &Input, start: (usize, usize)) -> DistMap {
        let mut dist_map: DistMap = vec![vec![std::usize::MAX; input.N]; input.N];
        let mut q = VecDeque::new();
        q.push_back(start);
        dist_map[start.0][start.1] = 0;
        while let Some((i, j)) = q.pop_front() {
            for dir in 0..4_usize {
                let (di, dj) = DIJ[dir];
                let (ni, nj) = (i + di, j + dj);
                if ni >= input.N || nj >= input.N {
                    continue;
                }
                if (di == 0 && input.v[i][min(j, nj)] == '0')
                    || (dj == 0 && input.h[min(i, ni)][j] == '0')
                {
                    if dist_map[ni][nj] > dist_map[i][j] + 1 {
                        dist_map[ni][nj] = dist_map[i][j] + 1;
                        q.push_back((ni, nj));
                    }
                }
            }
        }
        dist_map
    }

    pub fn move_between_two_points(
        input: &Input,
        from: (usize, usize),
        to: (usize, usize),
        output: &mut Output,
        acts_map: &mut BTreeMap<((usize, usize), (usize, usize)), Vec<char>>,
        current_day: &mut usize,
        prev_day: &mut Vec<Vec<usize>>,
        remains: &mut BTreeSet<(usize, usize)>,
        dist_maps: &Vec<Vec<DistMap>>,
    ) {
        if let Some(act) = acts_map.get(&(from, to)) {
            output.add(&act);
            regist_prev_day(from, &act, current_day, prev_day);
            remove_remains_from_actions(from, &act, remains)
        } else {
            let act = move_by_bfs(input, from, to, acts_map, current_day, prev_day, dist_maps);
            output.add(&act);
            regist_prev_day(from, &act, current_day, prev_day);
            remove_remains_from_actions(from, &act, remains)
        }
    }

    pub fn move_around_pos_set(
        input: &Input,
        current_day: &mut usize,
        last_pos: &mut (usize, usize),
        acts_map: &mut BTreeMap<((usize, usize), (usize, usize)), Vec<char>>,
        pos_set: &mut BTreeSet<(usize, usize)>,
        output: &mut Output,
        prev_day: &mut Vec<Vec<usize>>,
        remains: &mut BTreeSet<(usize, usize)>,
        dist_maps: &Vec<Vec<DistMap>>,
    ) {
        let entry_pos = *last_pos;
        let mut pos_vec: Vec<(usize, usize)> = vec![];
        let mut i = entry_pos.0;
        let mut j = entry_pos.1;
        while pos_set.len() > 0 {
            let mut q = VecDeque::new();
            q.push_back((i, j));
            let mut pos_min = (std::usize::MAX, std::usize::MAX);
            let mut min_dist = std::usize::MAX;
            for &(ni, nj) in pos_set.iter() {
                let dist_map = &dist_maps[i][j];
                let dist = dist_map[ni][nj];
                if dist < min_dist {
                    min_dist = dist;
                    pos_min = (ni, nj);
                }
            }
            pos_vec.push(pos_min);
            pos_set.remove(&pos_min);
            i = pos_min.0;
            j = pos_min.1;
        }

        let mut actions = Vec::<char>::new();
        for &(i, j) in pos_vec.iter() {
            let start = *last_pos;
            let goal = (i, j);
            if start == goal {
                continue;
            }

            let from = start;
            let to = goal;
            move_between_two_points(
                input,
                from,
                to,
                output,
                acts_map,
                current_day,
                prev_day,
                remains,
                dist_maps,
            );

            *last_pos = goal;
        }
    }

    pub fn regist_prev_day(
        entry_pos: (usize, usize),
        actions: &Vec<char>,
        current_day: &mut usize,
        prev_day: &mut Vec<Vec<usize>>,
    ) {
        let mut pos = entry_pos;
        prev_day[pos.0][pos.1] = *current_day;
        for act in actions.iter() {
            match act {
                'R' => pos.1 += 1,
                'L' => pos.1 -= 1,
                'D' => pos.0 += 1,
                'U' => pos.0 -= 1,
                _ => unreachable!(),
            }
            *current_day += 1;
            prev_day[pos.0][pos.1] = *current_day;
        }
    }

    pub fn move_by_bfs(
        input: &Input,
        from: (usize, usize),
        to: (usize, usize),
        acts_map: &mut BTreeMap<((usize, usize), (usize, usize)), Vec<char>>,
        current_day: &usize,
        prev_day: &mut Vec<Vec<usize>>,
        dist_maps: &Vec<Vec<DistMap>>,
    ) -> Vec<char> {
        let mut q = VecDeque::new();
        q.push_back(from);
        let mut prev = vec![vec![(0, 0); input.N]; input.N];
        let dist_and_sum_a = (std::usize::MAX, 0);
        let mut min_dist = vec![vec![dist_and_sum_a; input.N]; input.N];
        min_dist[from.0][from.1] = (0, 0);
        while let Some((i, j)) = q.pop_front() {
            for dir in 0..4_usize {
                let (di, dj) = DIJ[dir];
                let (ni, nj) = (i + di, j + dj);
                if ni >= input.N || nj >= input.N {
                    continue;
                }

                if (di == 0 && input.v[i][min(j, nj)] == '0')
                    || (dj == 0 && input.h[min(i, ni)][j] == '0')
                {
                    let dist = min_dist[i][j].0 + 1;
                    if dist > dist_maps[from.0][from.1][ni][nj] {
                        continue;
                    }
                    let days = *current_day - prev_day[ni][nj] + dist;
                    let sum_a = min_dist[i][j].1 + input.d[ni][nj] * days as i64;

                    let mut can_update = false;

                    // 最短経路を更新
                    if min_dist[ni][nj].0 > dist {
                        can_update = true;
                    } else if min_dist[ni][nj].0 == dist {
                        // 同じ距離の場合は、sum_aが大きい方を優先
                        if min_dist[ni][nj].1 < sum_a {
                            can_update = true;
                        }
                    }

                    if can_update {
                        min_dist[ni][nj].0 = dist;
                        min_dist[ni][nj].1 = sum_a;
                        prev[ni][nj] = (i, j);
                        q.push_back((ni, nj));

                        if (ni, nj) == to {
                            break;
                        }
                    }
                }
            }
        }

        let mut sum_d = 0;
        let mut path = VecDeque::new();
        let mut cur = to;
        while cur != from {
            let (i, j) = cur;
            let (pi, pj) = prev[i][j];

            if pi == i {
                if pj < j {
                    path.push_front('R');
                } else {
                    path.push_front('L');
                }
            } else {
                if pi < i {
                    path.push_front('D');
                } else {
                    path.push_front('U');
                }
            }

            cur = prev[i][j];

            sum_d += input.d[i][j];
        }

        let actions = Vec::from(path);

        if actions.len() != 0 && (sum_d / actions.len() as i64) > 500 {
            acts_map.insert((from, to), actions.clone());
        }

        actions
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
        pub const LIMIT: f64 = 1.9;
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
        // for i in 0..input.N {
        //     for j in 0..input.N {
        //         average[i][j] = sum[i][j] as f64 / L as f64;
        //     }
        // }
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
