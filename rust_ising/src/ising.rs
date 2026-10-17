use std::env;
include!(concat!(env!("OUT_DIR"),"/dim.rs"));
include!(concat!(env!("OUT_DIR"),"/weights.rs"));
use rand::{SeedableRng, RngExt};
use rand_pcg::Pcg64Mcg;
use bitvec::prelude::*;
use anyhow::Result;

macro_rules! weight_coord_base {
  ($i:expr, $j:expr, $deg:expr) => { ($deg * $i + $j) };
}

const RIGHT_MASK:u64 = (u32::MAX) as u64;
const SPIN_BIT:u64 = 1<<32;

pub type Bits = BitSlice<usize, LocalBits>;

pub trait Ising{
  fn new(sizes_v:Vec<usize>, seed:u64, magnetic_field_v:Vec<f64>, mu:f64,
    deg:usize) -> Self;
  fn init_cost(&mut self);
  fn init_spins(&mut self);
  fn n_points(&self) -> usize;
  fn cost_diff(&self, node:usize) -> f64;
  fn flip_spin(&mut self, node:usize, delta:f64);
  fn cost(&self) -> f64;
  fn deg(&self) -> usize;
  fn set_edges(&mut self, weights:Vec<f64>);
  fn set_spins(&mut self, spins:BitVec);
  fn get_size(&self, i:usize) -> usize;
  fn get_sizes(&self) -> &[usize];
  fn get_rng(&mut self) -> &mut Pcg64Mcg;
  fn set_starting_configs(&mut self,
    starting_configs:Option<Vec<StartingConfig>>);
  fn cluster_cost_diff(&self, cluster:&Bits, center:usize,
    ground_center_bit:bool) -> f64;
  fn config_eq(&self, other:&BitSlice) -> bool;
  fn config_to_bitvec(&self) -> BitVec;
/* Generic Functions */
  #[inline(always)]
  fn dim(&self) -> usize{
    return self.deg()/2;
  }
  fn to_coord(&self, node:usize) -> Vec<usize>{
    let mut node_copy = node;
    let dim = self.dim();
    let mut coord = vec![0; dim];
    for i in 0..dim{
      coord[i] = node_copy % self.get_size(i);
      node_copy /= self.get_size(i);
      }
    coord
    }

  fn from_coord(&self, coord: &[usize]) -> usize{
    debug_assert_eq!(coord.len(), self.get_sizes().len());
    coord.iter()
      .zip(self.get_sizes().iter())
      .fold((0,1), |(node, base), (&pos, &dim)|{
        (node + pos * base, base * dim)
      })
    .0
  }
  fn neighbours(&self, node:usize) -> Vec<usize>{
    let mut neighbours = Vec::<usize>::new();
    let mut node_coord = self.to_coord(node);
    for i in 0..self.dim(){
      let size_i = self.get_size(i);
      if size_i > 1 {
        node_coord[i] = (node_coord[i] + 1) % size_i;
        neighbours.push(self.from_coord(&node_coord));
        node_coord[i] = (node_coord[i] + size_i -2) % size_i;
        neighbours.push(self.from_coord(&node_coord));
        node_coord[i] = (node_coord[i] + 1) % size_i
      } else {
        neighbours.push(node);
        neighbours.push(node);
      }
    }
    neighbours
  }
  fn anneal(&mut self, temp:f64){
    let n_points = self.n_points();
    let node:usize = self.get_rng().random_range(0..n_points);
    let delta = self.cost_diff(node);
    if delta < 0.0 || (-delta / temp).exp() > self.get_rng()
      .random_range(0.0..1.0){
      self.flip_spin(node, delta);
    }
  }
  fn inf_anneal(&mut self){
    let n_points = self.n_points();
    let node:usize = self.get_rng().random_range(0..n_points);
    let delta = self.cost_diff(node);
    self.flip_spin(node, delta);

  }
  fn cluster_cost_weighted<F: Fn(f64, f64) -> f64>(&mut self,
    temperature:f64,
    center:usize,
    cluster: &Bits, 
    ground_opt: Option<&Bits>,
    ground_center_opt:Option<bool>,
    weight_func:F
    )
    -> f64 {
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
    let energy_diff = self.cluster_cost_diff(&cluster, center, 
      ground_center_bit);
    let sign = if energy_diff > 0.0 { -1.0 } else { 1.0 };
    sign * cluster_weight * weight_func(energy_diff, temperature) * 
      (1.0 - (- energy_diff.abs() / temperature).exp())
  }

  fn cluster_cost(&mut self, 
    temperature: f64,
    center:usize,
    cluster: &Bits,
    ground_opt: Option<&Bits>,
    ground_center_opt:Option<bool>) -> f64{
    self.cluster_cost_weighted(
      temperature,
      center,
      cluster,
      ground_opt,
      ground_center_opt,
      |_, _| 1.0f64,
    )
  }
  fn low_temp_pert_weighted<F: Fn(f64, f64) -> f64>(
    &mut self,
    local_min_approx:BitVec, 
    temperature:f64,
    weight_func:F
    )->f64{
    self.set_spins(local_min_approx);
    (0..self.n_points()).map(|v| (-self.cost_diff(v).abs() / temperature).exp()
      * weight_func(self.cost_diff(v), temperature))
      .sum::<f64>() / self.n_points() as f64
  }
  fn low_temp_pert(&mut self, local_min_approx:BitVec, temperature:f64) -> f64{
    self.low_temp_pert_weighted(local_min_approx, temperature, |_,_| 1.0f64)
  }
  fn reinitialize(&mut self){
    self.init_spins();
    self.init_cost();
  }

}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum IsingRepresentation {Disjoint, Packed}

#[derive(Clone,Copy,Debug, Default)]
pub struct IsingEdge {
  pub neighbour: usize,
  pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct StartingConfig{
  pub config:BitVec,
  pub weight:f64
}

#[derive(Debug)]
pub struct IsingDisjoint {
  pub deg: usize,
  sizes: Box<[usize]>,
  pub n_points: usize,
  pub spins: BitVec,
  edges: Vec<IsingEdge>,
  magnetic_field: Box<[f64]>,
  pub cost: f64,
  starting_configs: Option<Vec<StartingConfig>>,
  random_number_generator: Pcg64Mcg
}
#[derive(Debug)]
struct WeightsAndSpinStates{
  deg:usize,
  stride:usize,
  buffer:Box<[u64]>
}

impl WeightsAndSpinStates{
  pub fn new(n:usize, deg:usize) -> Self{
    let stride = 2*deg + 2;
    WeightsAndSpinStates{
      deg,
      stride,
      buffer: vec![0u64; n*stride].into_boxed_slice() 
      }
    }
  #[inline] fn base(&self, v:usize) ->usize { v*self.stride }
  #[inline] 
  fn edge_slot(&self, v:usize, i:usize) -> usize{ self.base(v) + 2 + 2 * i }
  #[inline]
  fn neighbour_slot(&self, v:usize, i:usize) -> usize{
    self.base(v) + 3 + 2 * i
  }
  #[inline] 
  fn vertex_spin(&self,v:usize) -> bool{ self.buffer[self.base(v)] == 1 }
  #[inline]
  fn vertex_mag(&self,v:usize) -> f64 {
    f64::from_bits(self.buffer[self.base(v)+1])
  }
  #[inline]
  fn vertex_edge(&self, v:usize, i:usize) -> f64{ 
    f64::from_bits(self.buffer[self.edge_slot(v,i)])
    }
  #[inline]
  fn vertex_neighbour(&self, v:usize, i:usize) -> u32{
    (self.buffer[self.neighbour_slot(v,i)] & RIGHT_MASK) as u32
    }
  #[inline]
  fn vertex_neighbour_spin(&self, v:usize, i:usize) -> bool{
    (self.buffer[self.neighbour_slot(v,i)] >> 32) != 0
  }
  fn set_vertex_edge_weight(&mut self, v:usize, i:usize, weight:f64){
    self.buffer[self.edge_slot(v,i)] = f64::to_bits(weight);
  }
  fn set_vertex_edge_neighbour(&mut self, v:usize, i:usize, neighbour:usize){
    let slot = self.neighbour_slot(v,i);
    self.buffer[slot] = neighbour as u64 | (self.buffer[slot] & SPIN_BIT);
  }
  #[inline]
  fn flip_neighbour_spin_internal(&mut self, neighbour:u32, node:u32){
    for i in 0..self.deg{
      if self.vertex_neighbour(neighbour as usize, i) == node{
        let neighbour_slot_index = self.neighbour_slot(neighbour as usize, i);
        self.buffer[neighbour_slot_index] ^= SPIN_BIT;
        break;
      }
    }
  }
  fn set_neighbour_spin_internal(&mut self, neighbour:u32, node:u32, 
    spin:bool){
    for i in 0..self.deg{
      /* I am the neighbour of my neighbour */
      if self.vertex_neighbour(neighbour as usize, i) == node {
        let neighbour_slot_index = self.neighbour_slot(neighbour as usize,i);
        let slot = &mut self.buffer[neighbour_slot_index];
        *slot = (*slot & RIGHT_MASK) | ((spin as u64)<<32);
        break;
      }
    }
  }
  fn set_spin_weight_and_spin_states(&mut self, node:usize, spin:bool){
    let base = self.base(node);
    self.buffer[base]=spin as u64;
    for i in 0..self.deg{
      let neighbour = self.vertex_neighbour(node, i);
      self.set_neighbour_spin_internal(neighbour, node as u32, spin);
    }
  }
  fn set_magnetic_field_weights_and_spin_states
    (&mut self, v:usize, mag_field:f64, mu:f64){
    let base = self.base(v);
    let mag_field_weighted = mu * mag_field;
    self.buffer[base + 1] = f64::to_bits(mag_field_weighted);
  } 
  fn flip_spin_weights_and_spin_states(&mut self, v:usize){
    let base = self.base(v);
    self.buffer[base] ^= 1;
    for i in 0..self.deg{
      let neighbour = self.vertex_neighbour(v,i);
      self.flip_neighbour_spin_internal(neighbour, v as u32);
    }
  }
}
#[derive(Debug)]
pub struct IsingPacked {
  deg:usize,
  sizes:Box<[usize]>,
  weights_and_spin_states:WeightsAndSpinStates,
  cost:f64,
  starting_configs: Option<Vec<StartingConfig>>,
  random_number_generator: Pcg64Mcg,
  n_points:usize,
}

impl IsingPacked{
  pub fn new(
    sizes_v:Vec<usize>,
    seed:u64,
    magnetic_field:Vec<f64>,
    mu:f64,
    deg:usize) -> Self
  {
    let n_points = sizes_v.iter().product();
    let sizes = sizes_v.into_boxed_slice();
    let cost = 0.0;
    let magnetic_field = magnetic_field;
    let mut weights_and_spin_states = WeightsAndSpinStates::new(
      n_points, deg
    );
    for (node, field) in magnetic_field.iter().enumerate(){
      weights_and_spin_states.set_magnetic_field_weights_and_spin_states
        (node, *field, mu);
    }
    let random_number_generator=Pcg64Mcg::seed_from_u64(seed);
    IsingPacked{
      deg, sizes,
      weights_and_spin_states, cost,
      starting_configs: None, random_number_generator,
      n_points,
    }
  }
  #[inline]
  fn get_spin(&self, v:usize) -> bool{
    self.weights_and_spin_states.vertex_spin(v)
  }
  #[inline]
  fn get_neighbour_spin(&self, node:usize, neighbour_index:usize) -> bool{
    self.weights_and_spin_states.vertex_neighbour_spin(node, neighbour_index)
  }
  #[inline]
  fn get_edge(&self, v:usize, i:usize) -> f64 {
    self.weights_and_spin_states.vertex_edge(v,i)
  }
  #[inline]
  fn get_neighbour(&self, v:usize, i:usize) -> usize{
    self.weights_and_spin_states.vertex_neighbour(v,i) as usize
  }
  #[inline]
  fn get_magnetic_field(&self, v:usize) -> f64{
    self.weights_and_spin_states.vertex_mag(v)
  }
  fn set_spins_packed(&mut self, conf:&BitSlice){
    debug_assert_eq!(conf.len(), self.n_points);
    for (node, spin) in conf.iter().by_vals().enumerate(){
      self.weights_and_spin_states.set_spin_weight_and_spin_states(node, spin)
    }
  }
  fn flip_spin_packed(&mut self, node:usize, delta:f64){
    self.weights_and_spin_states.flip_spin_weights_and_spin_states(node);
    self.cost += delta;
  }
  fn init_cost_packed(&mut self){
    self.cost = (0..self.n_points)
      .fold(0.0, |cost, node| {
        let spin_home = self.get_spin(node);
        let interaction = (0..self.deg)
          .fold(0.0, |local_weight, neighbour_idx|{
            let neighbour = self.get_neighbour(node,neighbour_idx);
            let edge = self.get_edge(node, neighbour_idx);
            let sign = (1 - 2 * ((spin_home == self.get_spin(neighbour))
                as i32 )) as f64;
            eprintln!("weight {} {} = {}", node, neighbour, edge);
            eprintln!("neighbour spin = {}, my spin = {}", 
              self.get_spin(neighbour), spin_home);
            local_weight + sign * edge
          });
        cost
          + self.get_magnetic_field(node) * (1 - 2 * spin_home as i32) as f64
          + interaction / 2.0
    });
  }
  fn set_edges_packed(&mut self, weights:Vec<f64>) -> Result<()>{
    let deg = self.deg;
    let weight_coord = |i:usize, j:usize|
      weight_coord_base!(i, j, deg);
    if weights.len() != deg*self.n_points {
      anyhow::bail!(
        "ising.rs: number of weights supplied to set_edges_packed \
        was incorrect."
      );
    }
    for v in 0..self.n_points{
      let neighbours = self.neighbours(v);
      for i in 0..deg{
        let neighbour_i = neighbours[i];
        self.weights_and_spin_states
          .set_vertex_edge_neighbour(v, i, neighbour_i);
        let duplicate = (0..i)
          .any(|j| self.get_neighbour(v,j) == neighbour_i);
        let self_loop = neighbour_i == v;
        if self_loop || duplicate {
          continue;
        }
        self.weights_and_spin_states
          .set_vertex_edge_weight(v, i, weights[weight_coord(v,i)]);

      }
    }
    Ok(())
  }
  fn set_starting_configs_packed(&mut self,
    starting_config: Option<Vec<StartingConfig>>)
  {
    self.starting_configs=starting_config;
  }
  fn init_spins_packed(&mut self){
    let starting_configs = self.starting_configs.clone();
    let Some(configs) = starting_configs else {
      self.init_spins_unif();
      return;
    };
   let sum_floats = configs.iter()
     .fold(0.0, |acc, config| acc+config.weight);
    let bound = self.random_number_generator.random_range(0.0..sum_floats);
    let mut accumulated_weight = 0.0;
    let mut i = 0;
    while accumulated_weight + configs[i].weight < bound {
      accumulated_weight += configs[i].weight;
      i+=1;
    }
    self.set_spins_packed(&configs[i].config);
  }
  fn init_spins_unif(&mut self){
    let conf:BitVec = (0..self.n_points)
      .map(|_| self.random_number_generator.random_bool(0.5))
      .collect();
    self.set_spins_packed(&conf);
  }
  fn cost_diff_packed(&self, node:usize) -> f64{
    let old_spin = self.get_spin(node);
    let interaction_diff = (0..self.deg)
      .fold(0.0, |local_weight, neighbour_idx| {
        let edge_weight = self.get_edge(node, neighbour_idx);
        let neighbour_spin = self.get_neighbour_spin(node, neighbour_idx);
        let was_equal = (neighbour_spin == old_spin) as isize;
        local_weight + ((2 * was_equal - 1) as f64) * edge_weight
      });
    let mag_diff = (((2 * old_spin as isize) - 1) as f64) * 
      self.get_magnetic_field(node);
    2.0 * (interaction_diff + mag_diff)
  }
  fn cluster_cost_diff_packed(&self, cluster: &Bits, center:usize,
    ground_center_bit:bool) -> f64{
    let interaction_diff = (0..self.deg)
      .fold(0.0, |local_weight, neighbour_idx| {
        let edge_weight = self.get_edge(center, neighbour_idx);
        let neighbour_spin = cluster[neighbour_idx];
        let was_equal = (neighbour_spin == ground_center_bit) as isize;
        local_weight + ((2 * was_equal - 1) as f64) * edge_weight
      });
    let mag_diff = (((2 * ground_center_bit as isize) - 1) as f64) * 
      self.get_magnetic_field(center);
    2.0 * (interaction_diff + mag_diff)
  }
}
impl IsingDisjoint {
  fn new(
    sizes_v:Vec<usize>, 
    seed:u64,
    magnetic_field_v:Vec<f64>,
    mu:f64,
    deg:usize) -> Self{
    let sizes = sizes_v.into_boxed_slice();
    let n_points = sizes.iter().product();
    let cost = 0.0;
    let magnetic_field = magnetic_field_v
      .iter()
      .map(|mag| mag * mu)
      .collect::<Vec<f64>>()
      .into_boxed_slice();
    let spins = bitvec![0; n_points];
    /* This is slightly suboptimal, but we prefer to make it impossible *
     * for an IsingDisjoint to be in an impossible state.               */
    let edges = vec![IsingEdge::default(); n_points];
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
  fn set_edges_disjoint(&mut self, weights:Vec<f64>){
    let mut edges = Vec::<IsingEdge>::new();
    let weight_coord = |i: usize, j: usize| 
      weight_coord_base!(i, j, self.deg);
    for node in 0..self.n_points{
      let mut row_edges = Vec::new();
      let start = weight_coord(node, 0);
      let end = weight_coord(node, self.deg);
      let neighbours = self.neighbours(node);
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
  fn set_starting_configs_disjoint(&mut self, 
    starting_config: Option<Vec<StartingConfig>>){
    self.starting_configs = starting_config;
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
            eprintln!("weight {} {} = {}", node, edge.neighbour, edge.weight);
            eprintln!("neighbour spin = {}, my spin = {}", 
              self.spins[edge.neighbour], spin_home);
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
  fn set_spins_disjoint(&mut self, spins: BitVec){
    self.spins = spins;
  }

  fn cost_diff_disjoint(&self, node:usize) -> f64{
    let old_spin = self.spins[node];
  
    let interaction_diff = (0..self.deg)
      .fold(0.0,|local_weight, neighbour_idx|{
        let edge = self.get_edge(node, neighbour_idx);
        let neighbour_spin = self.spins[edge.neighbour];
        let was_equal = (neighbour_spin == old_spin) as isize;
        local_weight + ((2 * was_equal - 1) as f64) * edge.weight
    });

    let mag_diff = ((2 * (old_spin as i32) - 1) as f64) * 
      self.magnetic_field[node];
    2.0 * (interaction_diff + mag_diff)
  }
  fn cluster_cost_diff_disjoint(&self, cluster: &Bits, center:usize,
    ground_center_bit:bool) -> f64{
    let interaction_diff = (0..self.deg)
      .fold(0.0, |local_weight, neighbour_idx|{
        let edge = self.get_edge(center, neighbour_idx);
        let neighbour_spin = cluster[neighbour_idx];
        let was_equal = (neighbour_spin == ground_center_bit) as i32;
        local_weight + ((2 * was_equal - 1) as f64) * edge.weight
    });
    let mag_diff = ((2 * (ground_center_bit) as i32 - 1) as f64) * 
      self.magnetic_field[center];
    2.0 * interaction_diff * mag_diff
  }
  fn flip_spin_disjoint(&mut self, node:usize, delta:f64){
    self.cost = self.cost + delta;
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
  #[inline]
  fn n_points(&self) -> usize { self.n_points }
  #[inline]
  fn cost_diff(&self, node:usize) -> f64{ self.cost_diff_disjoint(node) }
  #[inline]
  fn config_eq(&self, other:&BitSlice) -> bool {
    self.spins.as_bitslice() == other
  }
  #[inline]
  fn config_to_bitvec(&self) -> BitVec { self.spins.clone() }
  #[inline]
  fn flip_spin(&mut self, node:usize, delta:f64){
    self.flip_spin_disjoint(node, delta); 
  }
  #[inline]
  fn cost(&self) -> f64{ self.cost }
  #[inline]
  fn deg(&self)->usize{ self.deg }
  #[inline]
  fn init_spins(&mut self){ self.init_spins_disjoint(); }
  #[inline]
  fn init_cost(&mut self){ self.init_cost_disjoint(); }
  #[inline]
  fn set_edges(&mut self, weights:Vec<f64>){ self.set_edges_disjoint(weights); }
  #[inline]
  fn set_spins(&mut self, spins:BitVec){ self.set_spins_disjoint(spins); }
  #[inline]
  fn set_starting_configs(
    &mut self,
    starting_config:Option<Vec<StartingConfig>>
    ){
    self.set_starting_configs_disjoint(starting_config);
  }
  #[inline]
  fn get_size(&self, i:usize) -> usize{ self.sizes[i] }
  #[inline]
  fn get_sizes(&self) -> &[usize]{ &self.sizes }
  #[inline]
  fn get_rng(&mut self) -> &mut Pcg64Mcg{ &mut self.random_number_generator }
  #[inline]
  fn cluster_cost_diff(&self, cluster: &Bits, center:usize,
    ground_center_bit:bool) -> f64{
    self.cluster_cost_diff_disjoint(cluster, center, ground_center_bit)
  }
  #[inline]
  fn new(sizes_v:Vec<usize>, seed:u64, magnetic_field_v:Vec<f64>, mu:f64,
    deg:usize) -> Self{
    IsingDisjoint::new(sizes_v, seed, magnetic_field_v, mu, deg)
  }
}

impl Ising for IsingPacked {
  #[inline]
  fn n_points(&self) -> usize { self.n_points }
  #[inline]
  fn cost_diff(&self, node:usize) -> f64 { self.cost_diff_packed(node) }
  #[inline]
  fn flip_spin(&mut self, node:usize, delta:f64) { 
    self.flip_spin_packed(node,delta);
  }
  #[inline]
  fn cost(&self) -> f64{ self.cost }
  #[inline]
  fn init_spins(&mut self){ self.init_spins_packed(); }
  #[inline]
  fn init_cost(&mut self){ self.init_cost_packed(); }
  #[inline]
  fn deg(&self) -> usize{ self.deg }
  #[inline]
  fn get_rng(&mut self) -> &mut Pcg64Mcg { &mut self.random_number_generator }
  #[inline]
  fn set_edges(&mut self, weights:Vec<f64>){
    self.set_edges_packed(weights).unwrap();
  }
  #[inline]
  fn set_spins(&mut self, spins:BitVec){ self.set_spins_packed(&spins); }
  #[inline]
  fn set_starting_configs(&mut self,
    starting_configs:Option<Vec<StartingConfig>>
  ){
    self.set_starting_configs_packed(starting_configs);
  }
  #[inline]
  fn get_size(&self, i:usize) -> usize{ self.sizes[i] }
  #[inline]
  fn get_sizes(&self) -> &[usize] { &self.sizes }
  #[inline]
  fn cluster_cost_diff(&self, cluster:&Bits, center:usize,
    ground_center_bit:bool) -> f64
  {
    self.cluster_cost_diff_packed(cluster, center, ground_center_bit)
  }
  #[inline]
  fn new(sizes_v:Vec<usize>, seed:u64, magnetic_field_v:Vec<f64>, mu:f64,
    deg:usize) -> Self{
    IsingPacked::new(sizes_v, seed, magnetic_field_v, mu, deg)
  }
  fn config_eq(&self, other:&BitSlice)->bool{
    other.len() == self.n_points
      && (0..self.n_points).all(|v| self.get_spin(v) == other[v])
  }
  fn config_to_bitvec(&self) -> BitVec {
    (0..self.n_points).map(|v| self.get_spin(v)).collect()
  }
}
