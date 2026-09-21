use std::env;
include!(concat!(env!("OUT_DIR"),"/dim.rs"));
include!(concat!(env!("OUT_DIR"),"/weights.rs"));
use rand::{SeedableRng, RngExt};
use rand_pcg::Pcg64Mcg;
use bitvec::prelude::*;

macro_rules! weight_coord_base {
  ($i:expr, $j:expr, $deg:expr) => { ($deg * $i + $j) };
}

pub type Bits = BitSlice<usize, LocalBits>;

pub trait Ising{
  fn init_cost(&mut self);
  fn init_spins(&mut self);
  fn n_points(&self) -> usize;
  fn cost_diff(&self, node:usize) -> f64;
  fn flip_spin(&mut self, node:usize);
  fn cost(&self) -> f64;
  fn deg(&self) -> usize;
  fn anneal(&mut self, temp:f64);
  fn inf_anneal(&mut self);
  fn config(&self)->&BitVec;
  fn neighbours(&mut self, node:usize) -> Vec<usize>;
  fn cluster_cost(&mut self, 
    temperature: f64,
    center:usize,
    cluster: &Bits,
    ground_opt: Option<&Bits>,
    ground_center_opt:Option<bool>) -> f64;
 }



enum Layout {Disjoint, Packed}

enum IsingRepresentations{
  Disjoint(IsingDisjoint),
// Packed(IsingPacked),
}

const fn dimension(deg:usize) -> usize {
  deg/2
}

#[derive(Clone,Copy,Debug)]
pub struct IsingEdge {
  pub neighbour: usize,
  pub weight: f64,
}

#[derive(Debug)]
pub struct StartingConfig{
  pub config:BitVec,
  pub weight:f64
}

#[derive(Debug)]
pub struct IsingDisjoint {
  pub deg: usize,
  sizes: Vec<usize>,
  pub n_points: usize,
  pub spins: BitVec,
  edges: Vec<IsingEdge>,
  magnetic_field:Vec<f64>,
  pub cost: f64,
  starting_configs: Option<Vec<StartingConfig>>,
  random_number_generator: Pcg64Mcg
}



impl IsingDisjoint {
  pub fn new(dim:usize, sizes:Vec<usize>, seed:u64) -> Self{
    let n_points = sizes.iter().product();
    let cost = 0.0;
    let magnetic_field= Vec::<f64>::new();
    let spins = BitVec::new();
    let edges = Vec::<IsingEdge>::new();
    let deg = dim*2;
    let random_number_generator=Pcg64Mcg::seed_from_u64(seed);
    IsingDisjoint{ deg, sizes, n_points, cost, magnetic_field, spins, edges,
    starting_configs: None,
    random_number_generator}
  } 
  #[inline(always)]
  fn get_edge(&self, vertex:usize, edge_index: usize) -> &IsingEdge{
    &self.edges[self.edge_index(vertex, edge_index)]
  }
  #[inline(always)]
  fn edge_index(&self, vertex:usize, edge_index:usize) -> usize{
    vertex * self.deg + edge_index
  }
  pub fn set_edges(&mut self, weights:Vec<f64>){
    let mut edges = Vec::<IsingEdge>::new();
    let weight_coord = |i: usize, j: usize| 
      weight_coord_base!(i, j, self.deg);
    for node in 0..self.n_points{
      let mut row_edges = Vec::new();
      let start = weight_coord(node, 0);
      let end = weight_coord(node, self.deg);
      let neighbours = self.neighbours_disjoint(node);
      for (&neighbour, &weight) in neighbours
          .iter()
          .zip(&weights[start..end]){
            if neighbour == node {
              row_edges.push(IsingEdge { neighbour, weight });
              continue;
            }
              if let Some(_) = row_edges.iter_mut()
                .find(|e| e.neighbour == neighbour) {
                  row_edges.push(IsingEdge { neighbour: node, weight: 0.0});
              } else {
                  row_edges.push(IsingEdge{ neighbour, weight })
                }
              }
        edges.extend(row_edges);
        }
    self.edges=edges;
  }
  pub fn set_mu(&mut self, mu:f64){
    /* NOTE: If you set mu multiple times, this will multiply the mus   *
     * together.                                                          */
    self.magnetic_field.iter_mut()
      .for_each(|h| *h *= mu);
  }
  pub fn set_magnetic_field(&mut self, magnetic_field: Vec<f64>){
    self.magnetic_field = magnetic_field;
  }
  pub fn set_starting_configs(&mut self, 
    starting_config: Option<Vec<StartingConfig>>){
    self.starting_configs = starting_config;
  }

  fn to_coord(&self, node:usize) -> Vec<usize>{
    let mut node_copy = node;
    let mut coord = vec![0; self.sizes.len()];
    for i in 0..self.sizes.len(){
      coord[i] = node_copy % self.sizes[i];
      node_copy /= self.sizes[i]
    }
    coord
  }
  fn from_coord(&self, coord:&[usize]) -> usize{
    coord.iter()
      .zip(self.sizes.iter())
      .fold((0,1), |(node, base), (&pos, &dim)| {
        (node + pos * base, base * dim)
      })
      .0
  }
  pub fn neighbours_disjoint(&self, node:usize) -> Vec<usize>{
    let mut neighbours = Vec::<usize>::new();
    let mut node_coord = self.to_coord(node);
    for i in 0..self.sizes.len(){
      if self.sizes[i] > 1{
        node_coord[i] = (node_coord[i] + 1) % self.sizes[i];
        neighbours.push(self.from_coord(&node_coord));
        node_coord[i] = (node_coord[i] + self.sizes[i]- 2) % self.sizes[i];
        neighbours.push(self.from_coord(&node_coord));
        node_coord[i] = (node_coord[i] + 1) % self.sizes[i];
      } else {
        neighbours.push(node);
        neighbours.push(node);
      }
    }
    neighbours
  }
  pub fn init_cost_disjoint(&mut self){
    self.cost = 
      (0..self.n_points)
      .fold(0.0, |cost, node| {
        let spin_home = self.spins[node];
        let interaction = (0..self.deg)
          .fold(0.0,|local_weight, neighbour_idx|{
            let edge = self.get_edge(node, neighbour_idx);
            let sign = (1 - 2 * ((self.spins[edge.neighbour] == spin_home) 
                as i32))as f64;
            local_weight + (sign * edge.weight)
          });

        let magnetic = ((1 - 2 * spin_home as i32) as f64 ) 
          * self.magnetic_field[node];
      
      cost + magnetic + interaction / 2.0
    });
  }
  pub fn init_spins_disjoint(&mut self){
    let Some(configs) = &self.starting_configs else {
      self.init_spins_unif();
      return;
    };
    let sum_floats = configs.iter()
     .fold(0.0, |acc, config| acc + config.weight);
    let bound = self.random_number_generator.random_range(0.0..sum_floats);
    let mut accumulated_weight = 0.0;
    let mut i = 0;
    while accumulated_weight + configs[i].weight < bound{
      accumulated_weight += configs[i].weight;
      i+=1;
    }
    self.spins = configs[i].config.clone();
  } 
  fn init_spins_unif(&mut self){
    self.spins = (0..self.n_points)
      .map(|_| self.random_number_generator.random_bool(0.5))
      .collect();
  }
  fn set_spins(&mut self, spins:BitVec){
    self.spins = spins;
  }

  fn cost_diff_disjoint(&self, node:usize) -> f64{
    let old_spin = self.spins[node];
  
    let interaction_diff = (0..self.deg)
      .fold(0.0,|local_weight, neighbour_idx|{
        let edge = self.get_edge(node, neighbour_idx);
        let neighbour_spin = self.spins[edge.neighbour];
        let was_equal = (neighbour_spin == old_spin) as i32;
        local_weight + ((2 * was_equal - 1) as f64) * edge.weight
    });

    let mag_diff = ((2 * (old_spin as i32) - 1) as f64) * 
      self.magnetic_field[node];
    2.0 * (interaction_diff + mag_diff)
  }
  fn anneal_disjoint(&mut self, temp:f64) {
    let node:usize = self.random_number_generator
      .random_range(0..self.n_points);
    let delta = self.cost_diff_disjoint(node);
    if delta < 0.0 || (-delta / temp).exp() > self.random_number_generator
      .random_range(0.0..1.0) {
      let flipped = !self.spins[node];
      self.spins.set(node, flipped);
      self.cost += delta;
      }  
  }
  fn inf_anneal_disjoint(&mut self){
    let node:usize = self.random_number_generator
      .random_range(0..self.n_points);
    let delta = self.cost_diff_disjoint(node);
    let flipped = !self.spins[node];
    self.spins.set(node, flipped);
    self.cost += delta;
  }
  fn cluster_cost_disjoint(&mut self,
    temperature:f64,
    center:usize,
    cluster: &Bits, 
    ground_opt: Option<&Bits>,
    ground_center_opt:Option<bool>
    )
    -> f64 {
    // e^{-|E_c|/T}(-1)^{E_c^-<E_c^+}
    let m:usize;
    let ground_center_bit = ground_center_opt.unwrap_or(false);
    if let Some(ground) = ground_opt{
      m = cluster.iter()
        .by_vals()
        .zip(ground.iter().by_vals())
        .fold(0, |omega, (clust_bit, ground_bit)|
          omega + (clust_bit ^ ground_bit) as usize
        );
    } else {
      m = cluster.iter()
        .by_vals()
        .fold(0, |omega, clust_bit| omega + clust_bit as usize);
    }
    let cluster_weight = integral_weight(m)
      .expect("ising.rs error: m is out of range");
    let interaction_diff = (0..self.deg)
      .fold(0.0, |local_weight, neighbour_idx|{
        let edge = self.get_edge(center, neighbour_idx);
        let neighbour_spin = cluster[neighbour_idx];
        let was_equal = (neighbour_spin == ground_center_bit) as i32;
        local_weight + ((2 * was_equal - 1) as f64) * edge.weight
    });
    let mag_diff = ((2 * (ground_center_bit) as i32 - 1) as f64) * 
      self.magnetic_field[center];
    let energy_diff = interaction_diff + mag_diff;
    let sign = if energy_diff > 0.0 { -1.0 } else { 1.0 };
    sign * cluster_weight *
      (1.0 - (-2.0 * energy_diff.abs() / temperature).exp())

  }
  fn flip_spin_disjoint(&mut self, node:usize){
    self.cost = self.cost + self.cost_diff_disjoint(node);
    let flipped = !self.spins[node];
    self.spins.set(node, flipped);
  }
}

const fn integral_weight(m:usize) -> Option<f64>{
   if m > 2*D+1{
     None
   } else {
    Some(INTEGRAL_WEIGHTS[m])
   }
}

impl Ising for IsingDisjoint {
  fn n_points(&self) -> usize {
    self.n_points
  }
  fn cost_diff(&self, node:usize) -> f64{
    self.cost_diff_disjoint(node)
  }
  fn anneal(&mut self, temp:f64){
    self.anneal_disjoint(temp);
  }
  fn inf_anneal(&mut self){
    self.inf_anneal_disjoint();
  }
  fn config(&self)->&BitVec{
    &self.spins
  }
  fn flip_spin(&mut self, node:usize){
    self.flip_spin_disjoint(node);
  }
  fn cost(&self) -> f64{
    self.cost
  }
  fn deg(&self)->usize{
    self.deg
  }
  fn cluster_cost(&mut self,
    temperature:f64,
    center:usize,
    cluster: &Bits,
    ground_opt:Option<&Bits>,
    ground_center_opt:Option<bool>) -> f64{
    self.cluster_cost_disjoint(temperature,
      center,
      cluster,
      ground_opt,
      ground_center_opt)
  }
  fn neighbours(&mut self, node:usize) -> Vec<usize>{
    self.neighbours_disjoint(node)
  }
  fn init_spins(&mut self){
    self.init_spins_disjoint();
  }
  fn init_cost(&mut self){
    self.init_cost_disjoint();
  }
}
