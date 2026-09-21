use std::io::{BufReader, BufRead, Write, BufWriter, Seek};
use std::path::Path;
use crate::ising::*;
use std::fs::File;
use bitvec::prelude::*;
use anyhow::{Context, Result};
use rand::{SeedableRng, prelude::*};
use rand_distr::{Normal, Uniform};
use rand_pcg::Pcg64Mcg;
use std::fs::OpenOptions;


 macro_rules! local_index { 
  ($rel_index:expr) => 
  {(
    if $rel_index % 2 == 0 { $rel_index + 1 } else { $rel_index - 1 }
    )}
 }
pub struct IsingSpecifiers{
  edges: Box<[f64]>,
  field: Box<[f64]>, 
  ext_field: f64
}

struct EdwardsAnderson<E, M, X>
  where 
    E:Distribution<f64>,
    M:Distribution<f64>,
    X:Distribution<f64>
    {
  edge_dist:E,
  magnetic_dist:M,
  ext_magnetic_dist:X,
}

pub enum IsingModels{
  EA{edge: Dist, mag: Dist, ext: Dist},
  AllDownAllOnesGroundState,
}

#[derive(Clone, Copy)]
pub enum Dist {
  Normal{ mu: f64, std_dev: f64 },
  Uniform{ low: f64, high: f64},
}

impl std::str::FromStr for Dist {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let (kind, params) = s.split_once(':').ok_or("expected KIND:A,B")?;
        let nums: Vec<f64> = params
            .split(',')
            .map(|p| p.trim().parse::<f64>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        match (kind.to_lowercase().as_str(), nums.as_slice()) {
            ("normal", [mu, sd]) => Ok(Dist::Normal { mu: *mu, std_dev: *sd }),
            ("uniform", [lo, hi]) => Ok(Dist::Uniform { low: *lo, high: *hi }),
            _ => Err(format!("bad distribution spec: {s}")),
        }
    }
}

trait GenerateWeights{
  fn generate_edge_weights<R: Rng + ?Sized>
    (&self, n:usize, d:usize, rng: &mut R) -> Box<[f64]>;
  fn generate_magnetic_fields<R: Rng + ?Sized>
    (&self, n:usize, rng: &mut R) -> Box<[f64]>;
  fn generate_ext_magnetic_field<R: Rng + ?Sized>
    (&self, rng: &mut R) -> f64;
}

impl<E, M, X> EdwardsAnderson<E, M, X>
where
    E: Distribution<f64>,
    M: Distribution<f64>,
    X: Distribution<f64>,
{
    fn new(edge_dist: E, magnetic_dist: M, ext_magnetic_dist: X) -> Self {
        Self { edge_dist, magnetic_dist, ext_magnetic_dist }
    }
}


pub enum Sampler{
  Normal(Normal<f64>),
  Uniform(Uniform<f64>)
}

impl Distribution<f64> for Sampler{
  fn sample <R: Rng + ?Sized>(&self, rng: &mut R) -> f64{
    match self{
      Sampler::Normal(d) => d.sample(rng),
      Sampler::Uniform(d) => d.sample(rng)
    }
  }
}

impl Dist {
  pub fn build(&self) -> Result<Sampler> {
    match *self {
      Dist::Normal { mu, std_dev } => Normal::new(mu, std_dev)
        .map(Sampler::Normal)
        .with_context(|| format!("invalid Normal(mu={mu}, std_dev={std_dev})")),
      Dist::Uniform { low, high } => Uniform::new(low, high)
        .map(Sampler::Uniform)
        .with_context(|| format!("invalid Uniform(low={low}, high={high})")),
    }
  }
}



impl<E, M, X> GenerateWeights for EdwardsAnderson<E,M,X>
where
    E: Distribution<f64>,
    M: Distribution<f64>,
    X: Distribution<f64>, 
  {
  fn generate_edge_weights<R: Rng + ?Sized>
    (&self, n:usize, d:usize, mut rng: &mut R)-> Box<[f64]>{
    let n_edges = d*n;
    (&self.edge_dist).sample_iter(&mut rng)
      .take(n_edges)
      .collect()
  }
  fn generate_magnetic_fields<R: Rng + ?Sized>
    (&self, n:usize, mut rng: &mut R) -> Box<[f64]>{
      (&self.magnetic_dist).sample_iter(&mut rng)
      .take(n)
      .collect()
  }
  fn generate_ext_magnetic_field<R: Rng + ?Sized>
    (&self, mut rng: &mut R) -> f64{
      (&self).ext_magnetic_dist.sample(&mut rng)
  }
}

pub fn from_ising_file_disjoint_simple(path: impl AsRef<Path>) -> 
(IsingDisjoint, f64, Option<BitVec>)
{
  let file = File::open(path).unwrap();
  let mut reader = BufReader::new(file);
  let mut in_section = false;
  let mut ising_instance:IsingDisjoint;
  let temp:f64;
  {
  let mut lines = (&mut reader).lines();
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$temp"){
      in_section = true;
      break
    }
  }
  assert!(in_section, "Could not find temperature");
  let line = lines.by_ref()
    .next()
    .expect("Could not find dimension and sizes")
    .expect("error reading line");
  let mut parser = line.split_whitespace();
  temp = parser
    .next().unwrap()
    .parse().unwrap();

  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$rng_seed"){
      in_section = true;
      break
    }
  }
  assert!(in_section, "Could not find rng_seed");
  let line = lines.by_ref()
    .next()
    .expect("Could not find seed")
    .expect("error reading line");
  let t = line.trim();
  let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X"))
    .unwrap_or(t);
  let seed = u64::from_str_radix(t, 16).unwrap();
  


  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$dim_sizes"){
      in_section = true;
      break
    }
  }
  assert!(in_section, "Could not find dimension and sizes");
  let line = lines.by_ref()
    .next()
    .expect("Could not find dimension and sizes")
    .expect("error reading line");
  let mut parser = line.split_whitespace();
  let dim:usize = parser
    .next().unwrap()
    .parse().unwrap();
  assert!(dim > 0, "Dimension cannot be 0, that is nonsensical");
  let mut sizes = Vec::<usize>::new();
  for _i in 0..dim{
    let length:usize = parser
      .next().expect("Error: Dimension and Sizes mismatch")
      .parse().expect("Error parsing sizes");
    sizes.push(length);
  }
  ising_instance = IsingDisjoint::new(dim, sizes, seed);
  in_section = false;
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$edge_weights_start") {
      in_section = true;
      break;
    }
  }
  assert!(in_section, "Could not find edge_weights");
  let mut weights = vec![0.0; ising_instance.deg*ising_instance.n_points];
  /* Difficult command time! */
  for node in 0..ising_instance.n_points{
    let mut i = 0;
    for neighbour in ising_instance.neighbours(node){
      if neighbour > node {
        let line = lines.by_ref()
          .next()
          .expect("Insufficient number of weights")
          .expect("Error reading line");
        let weight:f64 = line.split_whitespace()
          .next()
          .expect("Blank weight")
          .parse()
          .expect("Error reading line");
        weights[ising_instance.deg*node + i] = weight;
        weights[ising_instance.deg*neighbour + local_index!(i)] = weight;
      }
      i += 1;
    }
  }
  ising_instance.set_edges(weights);
  in_section=false;
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$mu") {
      in_section = true;
      break;
    }
  }
  assert!(in_section, "Could not find magnetic moment mu");
  ising_instance.set_mu(
    lines.by_ref()
      .next()
      .expect("Could not find magnetic moment mu")
      .expect("Error reading line below mu")
      .split_whitespace()
      .next()
      .expect("Blank magnetic moment")
      .parse::<f64>()
      .expect("Error reading magnetic moment mu")
  );
  in_section=false;
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$external_magnetic_field") {
      in_section = true;
      break;
    }
  }
  assert!(in_section, "Could not find external magnetic field section");
  ising_instance.set_magnetic_field(
    (0..ising_instance.n_points).map(|_| 
      lines.by_ref()
        .next()
        .expect("Error: Not enough entries in the external magnetic field")
        .expect("Error reading external magnetic field entries")
        .split_whitespace()
        .next()
        .expect("Blank magnetic field entry")
        .parse::<f64>()
        .expect("Error reading magnetic field entry")
    ).collect()
  );
  in_section=false;
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$starting_configs") {
      in_section = true;
      break;
    }
  }

  let hex_length = ising_instance.n_points.div_ceil(8);
  let starting_configs: Option<Vec<StartingConfig>>;
  if in_section {
    let mut starting_configs_base = Vec::<StartingConfig>::new();
    while let Some(line) = lines.next() {
      let line = line.expect("Error reading line");
      if line.trim().is_empty() {
        break;
      }
      let mut line = line.split_whitespace();
      let mut hex = Vec::<u8>::new();
      for _ in 0..hex_length{
        hex.push(
          u8::from_str_radix(
            line.next()
              .expect(
 "Error: Not enough entries for hex code of starting config"),
            16
          )
          .expect("Error: No integer detected")
        );
        }
      let chunk_size = std::mem::size_of::<usize>();
      let config_usize:Vec<usize> = hex.chunks(chunk_size)
        .map(|chunk| {
          chunk.iter()
            .enumerate()
            .fold(0usize, |acc, (i, &byte)|{
              acc | ((byte as usize) << i*8)
            })
          })
        .collect();
      let starting_config = usize_to_config(config_usize, 
        ising_instance.n_points);
      let prob = line.next()
        .expect("Could not find probability associated with starting 
          configuration")
        .parse::<f64>()
        .expect("Probability for starting configuration not a float");
      //DEBUG 
      #[cfg(debug_assertions)]
      {
        if prob > 1.0{
          eprintln!("ising_io.rs warning: Weight greater than 1.");
          }
      }
      starting_configs_base.push(StartingConfig{
        config:starting_config, weight:prob
      })
      }
    starting_configs=Some(starting_configs_base)
  }
  else {
    eprintln!("ising_io.rs warning: Starting Configurations not found. \
      Defaulting to random Starting Configurations");
    starting_configs = None;
  }
  ising_instance.set_starting_configs(starting_configs);
  }
  reader.rewind().expect("ising_io.rs error: error reading file");
  let hex_length = ising_instance.n_points.div_ceil(8);
  let mut lines = reader.lines();
  in_section = false;
  for line in lines.by_ref(){
    let line = line.unwrap();
    if line.starts_with("$ground_state_approx"){
      in_section = true;
      break
    }
  }
  let ground_state_approx:Option<BitVec> = if !in_section{
    None
  } else {
    let next_line = lines.next();
    match next_line {
      None => {
      eprintln!("ising_io.rs warning: ground_state_approx section is \
        present but has no entry");
      None
    } Some(line) => {
    let line = line.expect("ising_io.rs error: could not read line");
    if line.is_empty(){
      eprintln!("ising_io.rs warning: ground_state_approx section is \
        present but has no entry");
      None
    } else {
      let mut line = line.split_whitespace();
      let mut hex = Vec::<u8>::new();
      for _ in 0..hex_length{
      hex.push(
        u8::from_str_radix(
          line.next()
          .expect("Error: Not enough entries for hex code of ground state \
            config"),
          16
          ).expect("Error: ground_state not given in hex?")
        );
      }
      let chunk_size = std::mem::size_of::<usize>();
      let config_usize:Vec<usize> = hex.chunks(chunk_size)
        .map(|chunk|{
          chunk.iter()
            .enumerate()
            .fold(0usize, |acc, (i, &byte)|{
              acc | ((byte as usize) << i*8)
            })
        })
      .collect();
        Some(usize_to_config(config_usize, ising_instance.n_points))
        } 
      }
    }
  };

  
  ising_instance.init_spins();
  ising_instance.init_cost();
  (ising_instance, temp, ground_state_approx)
}

fn usize_to_config(uvec:Vec<usize>, n_points:usize)->BitVec{
    let mut bit_vec = BitVec::from_vec(uvec);
    bit_vec.truncate(n_points);
    bit_vec
}

pub fn write_ground_state(
  ground_state_string:String,
  path: impl AsRef<Path>
  ) -> Result<()>{
  let write_err = "ising_io.rs error: could not write to input file";
  let file = File::open(&path).unwrap();
  let reader = BufReader::new(file);
  let lines:Vec<String> = reader.lines()
    .collect::<std::io::Result<Vec<_>>>()
    .context("ising_io.rs: could not read input file")?;
  if lines.iter().any(|line| line.starts_with("$ground_state_approx")){
    anyhow::bail!("ising_io.rs error: ground_state_approx already exists. \
      remove this line from this input file before running this code"
      );
  }
  let insert_at:usize;
  let target_start_confs = lines.iter().position(|line| {
    line.starts_with("$starting_configs") 
  });
  if let Some(target) = target_start_confs{
    insert_at = lines[target+1..]
      .iter()
      .position(|line| {
        line.trim().is_empty()
      })
      .map(|i| target + 1 + i)
      .unwrap_or(lines.len());
  } else {
    let target_ext_field = lines.iter().position(|line| {
      line.starts_with("$external_magnetic_field")
    }).expect("ising_io.rs error: no external magnetic field section in \
    input file caught during ground state overwrite but not during the \
    reading?");
    insert_at = lines[target_ext_field+1..]
      .iter()
      .position(|line| {
        line.trim().is_empty()
      })
      .map(|i| target_ext_field + 1 + i)
      .unwrap_or(lines.len());
  }
  let file = File::create(&path)?;
  let mut writer = BufWriter::new(file);
  for line in &lines[..insert_at]{
    writeln!(writer, "{line}").context(write_err)?;
  }
  writeln!(writer).context(write_err)?;
  writeln!(writer, "$ground_state_approx").context(write_err)?;
  writeln!(writer, "{ground_state_string}").context(write_err)?;
  let rest = if insert_at < lines.len() && lines[insert_at].trim().is_empty() {
    &lines[insert_at+1..]
  } else {
    &lines[insert_at..]
  };

  for line in rest {
    writeln!(writer, "{line}").context(write_err)?;
  }
  writer.flush().context("ising_io.rs: could not flush input file")?;
  Ok(())
 }

pub fn bitvec_to_hex_string(bv: &BitVec) -> String{
  let bv_len = bv.len();
  bv.chunks(64)
   .map(|chunk| { chunk.iter()
     .by_vals()
     .enumerate()
     .fold(0u64, |n, (i, bit)| n | ((bit as u64) << i))
   })
   .flat_map(|num| num.to_le_bytes())
   .take((bv_len + 7) / 8)
   .map(|byte| format!("{:02x}", byte))
   .collect::<Vec<String>>()
   .join(" ")
}


pub fn generate_ising_file_specifiers(
  model:IsingModels,
  dim:usize,
  n:usize,
  seed: u64
  ) -> Result<IsingSpecifiers>{
   match model{ 
    IsingModels::EA { edge, mag, ext} => {
      let mut rng = Pcg64Mcg::seed_from_u64(seed);
      let ea = EdwardsAnderson::new(
        edge.build()?,
        mag.build()?,
        ext.build()?
      );
      let edges = ea.generate_edge_weights(n, dim, &mut rng);
      let field = ea.generate_magnetic_fields(n, &mut rng);
      let ext_field = ea.generate_ext_magnetic_field(&mut rng);
      Ok(IsingSpecifiers { edges, field, ext_field })
  }
  IsingModels::AllDownAllOnesGroundState => {
    Ok(IsingSpecifiers{
    edges: (0..n*dim)
      .map(|_| 1.0)
      .collect(),
    field: (0..n)
      .map(|_| -1.0)
      .collect(),
    ext_field: 1.0,
    })
    }
  }
}

pub fn generate_ising_file(
  path: impl AsRef<Path>,
  specs:IsingSpecifiers,
  temp: f64,
  rng_seed: u64,
  dim: usize,
  sizes: Box<[usize]>
  ) -> Result<()>
{
  let file = OpenOptions::new()
    .write(true)
    .create_new(true)
    .open(path)?;

  let mut writer = BufWriter::new(file);
  writeln!(writer, "$temp")?;
  writeln!(writer, "{temp}\n")?;
  writeln!(writer, "$rng_seed")?;
  writeln!(writer, "{:x}\n", rng_seed)?;
  writeln!(writer, "$dim_sizes")?;
  write!(writer, "{dim}")?;
  for size in sizes{
    write!(writer, " {size}")?;
  }
  writeln!(writer, "\n")?;
  writeln!(writer, "$edge_weights_start")?;
  for edge in specs.edges{
    writeln!(writer, "{edge}")?;
  }
  writeln!(writer,"")?;
  writeln!(writer,"$mu")?;
  writeln!(writer,"{}\n", specs.ext_field)?;
  writeln!(writer,"$external_magnetic_field")?;
  for magnet in specs.field{
    writeln!(writer, "{magnet}")?;
  }
  writer.flush()?;
  Ok(())
}
