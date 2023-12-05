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

use crate::{solver::SimpleDfs, Tools::compute_score};

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
        let mut rng: Mcg128Xsl64 = rand_pcg::Pcg64Mcg::new(890482);
        let mut cnt = 0 as usize; // 試行回数

        let mut output: Output = Output::new();

        let solve = "3";
        if solve == "1" {
            let mut simple_dfs = SimpleDfs::new(self.input.clone());
            let start = (0, 0);
            simple_dfs.run(&mut output, start);
        } else if solve == "2" {
        } else if solve == "3" {
            let start = (0, 0);

            let mut solve = solver::CleanAroundHighA::new(self.input.N);
            let mut current_day = 0;
            let mut current_pos = start;
            let max_clean_cnt = self.input.N * 4;
            let mut acts_map = BTreeMap::new();
            solve.run(
                &self.input,
                current_pos,
                &mut current_day,
                &mut output,
                max_clean_cnt,
                &mut acts_map,
            );
            //areas.debug_id_map();
        } else if solve == "4" {
        }

        output.submit();

        let (score, _) = Tools::compute_score(&self.input, &output);

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
        let best_score = if output.out.len() >= 100000 {
            -1
        } else {
            score
        };
        eprintln!("{} ", best_score);
        eprintln!("{} ", self.input.N);
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

    fn remove(&self, output: &mut Self, rng: &mut Mcg128Xsl64) {
        // https://atcoder.jp/contests/ahc014/submissions/35567589 L558
    }

    fn submit(&self) {
        println!("{}", self.out.iter().collect::<String>());
    }
}

mod solver {
    use std::vec;

    use rand::seq::index;

    use super::*;

    pub fn compute_clearness(input: &Input, output: &Output) -> i64 {
        let mut clearness = 0;
        let mut pos = (0, 0);
        for &c in output.out.iter() {
            match c {
                'R' => pos.1 += 1,
                'L' => pos.1 -= 1,
                'D' => pos.0 += 1,
                'U' => pos.0 -= 1,
                _ => unreachable!(),
            }
            clearness += input.d[pos.0][pos.1];
        }
        clearness
    }
    #[derive(Clone, Debug)]
    pub struct Area {
        id: usize,
        center: (usize, usize),
        points: BTreeSet<(usize, usize)>,
        length: usize,
        d_ave: i64,
        d_max: i64,
        d_min: i64,
        d_sum: i64,
        prev_day: usize, // 前回掃除した日
        N: usize,
    }

    impl Area {
        pub fn new(id: usize, center: (usize, usize), length: usize, N: usize) -> Self {
            let mut points = BTreeSet::new();
            points.insert(center);
            Area {
                id,
                center,
                points,
                length,
                d_ave: 0,
                d_max: std::i64::MIN,
                d_min: std::i64::MAX,
                d_sum: 0,
                prev_day: 0,
                N,
            }
        }

        pub fn compute_d(&mut self, input: &Input) {
            for (i, j) in self.points.iter() {
                self.d_sum += input.d[*i][*j];
                self.d_max = self.d_max.max(input.d[*i][*j]);
                self.d_min = self.d_min.min(input.d[*i][*j]);
            }
            self.d_ave = self.d_sum / self.points.len() as i64;
        }

        pub fn get_area_a(&self, current_day: usize) -> i64 {
            let diff = (current_day - self.prev_day) as i64;
            let area_a = self.d_ave * diff;
            area_a
        }

        pub fn clean(
            &mut self,
            input: &Input,
            current_day: &mut usize,
            last_pos: &mut (usize, usize),
        ) -> Vec<char> {
            let mut has_seen = BTreeSet::new();
            let mut actions = Vec::<char>::new();
            let pre_act = 'X';
            self.dfs(
                input,
                pre_act,
                self.center,
                &mut has_seen,
                &mut actions,
                last_pos,
            );
            *current_day += actions.len();
            self.prev_day = *current_day;
            actions
        }

        fn dfs(
            &self,
            input: &Input,
            pre_act: char,
            current_pos: (usize, usize),
            has_seen: &mut BTreeSet<(usize, usize)>,
            actions: &mut Vec<char>,
            last_pos: &mut (usize, usize),
        ) {
            has_seen.insert(current_pos);
            if has_seen.len() == self.points.len() {
                *last_pos = current_pos;
                return;
            }

            let mut dir_ids = vec![];
            match pre_act {
                'R' => dir_ids = vec![1, 0, 3],
                'D' => dir_ids = vec![2, 1, 0],
                'L' => dir_ids = vec![3, 2, 1],
                'U' => dir_ids = vec![0, 3, 2],
                _ => dir_ids = vec![0, 1, 2, 3],
            };

            for dir in dir_ids {
                //for dir in 0..4_usize {
                let (di, dj) = DIJ[dir];
                let (ni, nj) = (current_pos.0 + di, current_pos.1 + dj);
                if ni >= self.N || nj >= self.N {
                    continue;
                }

                if has_seen.contains(&(ni, nj)) {
                    continue;
                }
                if !self.points.contains(&(ni, nj)) {
                    continue;
                }

                let act = DIR.chars().nth(dir).unwrap();
                if (di == 0 && input.v[current_pos.0][min(current_pos.1, nj)] == '0')
                    || (dj == 0 && input.h[min(current_pos.0, ni)][current_pos.1] == '0')
                {
                    actions.push(act);
                } else {
                    continue;
                }

                self.dfs(input, act, (ni, nj), has_seen, actions, last_pos);
                if has_seen.len() == self.points.len() {
                    return;
                }
                let c = DIR.chars().nth((dir + 2) % 4).unwrap();
                actions.push(c);
            }
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
    ) {
        let entry_pos = *last_pos;

        // areaのaの高い順にソート
        let mut a_ranking = vec![];
        for i in 0..input.N {
            for j in 0..input.N {
                let diff = (*current_day - prev_day[i][j]) as i64;
                let d = input.d[i][j];
                let a = d * diff;
                a_ranking.push((a, (i, j)));
            }
        }
        a_ranking.sort();
        a_ranking.reverse();

        let cnt_max = a_ranking.len() / 3;
        let mut cnt = 0;
        let mut pos_set = BTreeSet::new();
        for &(a, (i, j)) in a_ranking.iter() {
            cnt += 1;

            pos_set.insert((i, j));

            if input.d[i][j] == 0 {
                break;
            }

            if input.d[i][j] < 200 && cnt >= cnt_max {
                break;
            }
        }

        // entry_posから近い点に移動。その点からさらに近い点に移動。を繰り返す
        let mut actions = move_around_pos_set(input, current_day, last_pos, acts_map, &mut pos_set);
        remove_remains_from_actions(entry_pos, &actions, remains);
        // 通過した点のprev_dayを更新
        regist_prev_day(entry_pos, &actions, current_day, prev_day);
        output.add(&actions);
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

    #[derive(Clone, Debug)]
    pub struct CleanAroundHighA {}

    impl CleanAroundHighA {
        pub fn new(N: usize) -> Self {
            CleanAroundHighA {}
        }

        pub fn run(
            &mut self,
            input: &Input,
            mut current_pos: (usize, usize),
            current_day: &mut usize,
            output: &mut Output,
            max_clean_cnt: usize,
            acts_map: &mut BTreeMap<((usize, usize), (usize, usize)), Vec<char>>,
        ) {
            let mut remains = BTreeSet::new();
            for i in 0..input.N {
                for j in 0..input.N {
                    remains.insert((i, j));
                }
            }

            let mut prev_day = vec![vec![0; input.N]; input.N];

            for _ in 0..max_clean_cnt {
                if output.out.len() >= (100000 - (remains.len() + 2) * input.N * input.N) {
                    //eprintln!("{:?} ", output.out.len());
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
                );
            }

            let entry_pos = current_pos;
            let actions =
                move_around_pos_set(input, current_day, &mut current_pos, acts_map, &mut remains);
            regist_prev_day(entry_pos, &actions, current_day, &mut prev_day);
            output.add(&actions);

            let start = current_pos;
            let goal = (0, 0);
            if let Some(act) = acts_map.get(&(start, goal)) {
                output.add(&act);
            } else {
                let act = compute_path_with_bfs(input, start, goal);
                output.add(&act);
                acts_map.insert((start, goal), act.clone());
            }
        }
    }

    pub fn move_around_pos_set(
        input: &Input,
        current_day: &mut usize,
        last_pos: &mut (usize, usize),
        acts_map: &mut BTreeMap<((usize, usize), (usize, usize)), Vec<char>>,
        pos_set: &mut BTreeSet<(usize, usize)>,
    ) -> Vec<char> {
        // entry_posから近い点をbfsで探し移動。その点からさらに近い点に移動。を繰り返す
        let mut pos_vec = vec![];
        let mut i = last_pos.0;
        let mut j = last_pos.1;
        while pos_set.len() > 0 {
            let mut q = VecDeque::new();
            q.push_back((i, j));
            let mut has_seen = BTreeSet::new();
            has_seen.insert((i, j));
            let mut dist = vec![vec![std::usize::MAX; input.N]; input.N];
            dist[i][j] = 0;
            let mut pos_min = (std::usize::MAX, std::usize::MAX);
            'x: while let Some((i, j)) = q.pop_front() {
                for dir in 0..4_usize {
                    let (di, dj) = DIJ[dir];
                    let (ni, nj) = (i + di, j + dj);
                    if ni >= input.N || nj >= input.N {
                        continue;
                    }
                    if has_seen.contains(&(ni, nj)) {
                        continue;
                    }
                    if (di == 0 && input.v[i][min(j, nj)] == '0')
                        || (dj == 0 && input.h[min(i, ni)][j] == '0')
                    {
                        q.push_back((ni, nj));
                        has_seen.insert((ni, nj));
                        dist[ni][nj] = dist[i][j] + 1;
                        if pos_set.contains(&(ni, nj)) {
                            pos_min = (ni, nj);
                            break 'x;
                        }
                    }
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
            let from_to = (start, goal);
            if let Some(act) = acts_map.get(&from_to) {
                for &c in act.iter() {
                    actions.push(c);
                }
            } else {
                let act = compute_path_with_bfs(input, start, goal);
                for &c in act.iter() {
                    actions.push(c);
                }
                acts_map.insert(from_to, act.clone());
            }

            *last_pos = goal;
        }
        actions
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

    pub fn compute_d_ranking(input: &Input) -> Vec<(i64, (usize, usize))> {
        let mut d_ranking = vec![];
        for i in 0..input.N {
            for j in 0..input.N {
                d_ranking.push((input.d[i][j], (i, j)));
            }
        }
        d_ranking.sort();
        d_ranking.reverse();
        d_ranking
    }

    pub struct SimpleDfs {
        input: Input,
    }

    impl SimpleDfs {
        pub fn new(input: Input) -> Self {
            SimpleDfs { input }
        }

        pub fn run(self, output: &mut Output, start: (usize, usize)) {
            let mut has_seen = vec![vec![false; self.input.N]; self.input.N];

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

    pub struct DfsUntilWholeCleaning {
        input: Input,
    }

    impl DfsUntilWholeCleaning {
        pub fn new(input: Input) -> Self {
            DfsUntilWholeCleaning { input }
        }

        pub fn run(self, output: &mut Output, start: (usize, usize)) -> (usize, usize) {
            let mut has_seen = vec![vec![false; self.input.N]; self.input.N];
            let mut has_seen_cnt = 0;
            let mut goal_point = (std::usize::MAX, std::usize::MAX);

            self.dfs(
                start,
                &mut has_seen,
                output,
                &mut has_seen_cnt,
                &mut goal_point,
            );

            goal_point
        }

        fn dfs(
            &self,
            (i, j): (usize, usize),
            has_seen: &mut Vec<Vec<bool>>,
            output: &mut Output,
            has_seen_cnt: &mut usize,
            goal_point: &mut (usize, usize),
        ) {
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
                    *has_seen_cnt += 1;
                    if (*has_seen_cnt == self.input.N * self.input.N - 1) {
                        *goal_point = (ni, nj);
                    }

                    self.dfs((ni, nj), has_seen, output, has_seen_cnt, goal_point);
                    if (*has_seen_cnt == self.input.N * self.input.N - 1) {
                        return;
                    }
                    let c = DIR.chars().nth((dir + 2) % 4).unwrap();
                    output.out.push(c);
                }
            }
        }
    }

    pub fn compute_path_with_bfs(
        input: &Input,
        start: (usize, usize),
        goal: (usize, usize),
    ) -> Vec<char> {
        let mut q = VecDeque::new();
        q.push_back(start);
        let mut prev = vec![vec![(0, 0); input.N]; input.N];
        let mut min_dist = vec![vec![std::usize::MAX; input.N]; input.N];
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
                    // 最短経路を更新
                    if min_dist[ni][nj] > min_dist[i][j] + 1 {
                        min_dist[ni][nj] = min_dist[i][j] + 1;
                        prev[ni][nj] = (i, j);
                        q.push_back((ni, nj));

                        if (ni, nj) == goal {
                            break;
                        }
                    }
                }
            }
        }

        let mut path = VecDeque::new();
        let mut cur = goal;
        while cur != start {
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
        }

        path.into_iter().collect()
    }

    pub fn get_back_path(path: &Vec<char>) -> Vec<char> {
        let mut back_path = vec![];
        for &c in path.iter().rev() {
            let c = match c {
                'R' => 'L',
                'L' => 'R',
                'U' => 'D',
                'D' => 'U',
                _ => unreachable!(),
            };
            back_path.push(c);
        }
        back_path
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
