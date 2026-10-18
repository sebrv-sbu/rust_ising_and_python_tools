/* our import */
use crate::ising::*;
use crate::anneal::*;
use crate::greedy::*;

/* std */
use std::io::{BufReader, BufRead, Write, BufWriter, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::fs::File;
use std::fs::OpenOptions;
use std::str::FromStr;
use std::error::Error;

/* rand */
use rand::{SeedableRng, prelude::*};
use rand_distr::{Normal, Uniform};
use rand_pcg::Pcg64Mcg;

/* other */
use anyhow::{ensure, bail, Context, Result};
use bitvec::prelude::*;

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

const LOCAL_MIN_MISSING_ERR: &str = "No local minimum \
    supplied in file. Cannot compute low temperature perturbation. Consider \
    using the ground-state-approx mode with local minimum (-l) and modify \
    input file (-w) enabled";
const GLOBAL_MIN_MISSING_ERR: &str = "No global minimum \
    supplied in file. Cannot compute low temperature perturbation. Consider \
    using the ground-state-approx mode with modify input file (-w) enabled";

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

fn find_section<R:BufRead + Seek>(
  reader: &mut R,
  header: &str) -> Result<bool>{
  reader.seek(SeekFrom::Start(0))?; 
  let mut line = String::new();
  loop{
    line.clear();
    if reader.read_line(&mut line)? == 0 {
      return Ok(false);
    }
    if line.starts_with(header) {
      return Ok(true);
    }
  }
}

fn parse_line<T, R>(reader: &mut R, what:&str)->Result<Vec<T>>
where
  R:BufRead,
  T:FromStr,
  T::Err: Error + Send + Sync + 'static,
{
  let mut line = String::new();
  let n = reader.read_line(&mut line).with_context(||format!(
    "ising_io.rs: error reading {what}"))?;
  if n == 0 {
    bail!("ising_io.rs: unexpected end of file while reading {what}");
  }
  line.split_whitespace()
    .map(|tok| tok.parse::<T>()
      .with_context(|| format!("bad {what} token {tok:?}")))
    .collect()
}
#[inline]
fn read_single_entry<T,R>(reader:&mut R, what:&str) -> Result<T>
where
  R:BufRead,
  T:FromStr + Clone,
  T::Err: Error + Send + Sync + 'static,
{
  let l = parse_line::<T, R>(reader, what)
    .into_iter()
    .next()
    .with_context(|| format!("ising_io.rs: missing {what}"))?;
  Ok(l[0].clone())
}
fn read_weights<I, R>(reader: &mut R, ising_instance:&I)
  -> Result<Vec<f64>>
where 
  I:Ising,
  R:BufRead,
{
  let mut lines = reader.by_ref().lines();
  let mut weights = vec![0.0; ising_instance.deg()*ising_instance.n_points()];
  for node in 0..ising_instance.n_points(){
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
        weights[ising_instance.deg()*node + i] = weight;
        weights[ising_instance.deg()*neighbour + local_index!(i)] = weight;
      }
      i += 1;
    }
  }
  Ok(weights)
}
fn read_starting_configs<R:BufRead, I:Ising>(reader:&mut R, 
  ising_instance: &I) -> 
Result<Option<Vec<StartingConfig>>>
{
  let hex_length = ising_instance.n_points().div_ceil(8);
  let mut lines = reader.by_ref().lines();
  let mut starting_configs_base = Vec::<StartingConfig>::new();
  while let Some(line) = lines.next() {
    let line = line.with_context(||"ising_io.rs: starting_configs: error \
      reading line")?;
    if line.trim().is_empty() {
      break;
    }
    let mut line = line.split_whitespace();
    let mut hex = Vec::<u8>::new();
    for _ in 0..hex_length{
      hex.push(
        u8::from_str_radix(
          line.next().with_context(||"ising_io.rs: starting_configs: \
            Not enough entries for hex code of starting config")?,16)
        .with_context(|| format!("ising_io.rs: starting_configs: \
            No integer detected"))?
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
      ising_instance.n_points());
    let prob = line.next()
      .with_context(|| "ising_io.rs: starting_configs: ould not find \
        probability associated with starting configuration")?
      .parse::<f64>()
      .with_context(|| "ising_io.rs: starting_configs: probability for \
        starting configuration not a float")?;
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
  Ok(Some(starting_configs_base))
}

fn read_state<R:BufRead, I:Ising>(reader:&mut R, ising_instance:&I,
  state_type:&str)->Result<Option<BitVec>>{
  let hex_length = ising_instance.n_points().div_ceil(8);
  let mut lines = reader.by_ref().lines();
  let next_line = lines.next();
  match next_line {
    None => {
    eprintln!("ising_io.rs warning: {state_type} section is \
      present but has no entry");
    Ok(None)
  } Some(line) => {
  let line = line.with_context(||format!("ising_io.rs: error reading \
      {state_type} line"))?;
  if line.is_empty(){
    eprintln!("ising_io.rs warning: {state_type} section is \
      present but has no entry");
    Ok(None)
  } else {
    let mut line = line.split_whitespace();
    let mut hex = Vec::<u8>::new();
    for _ in 0..hex_length{
    hex.push(
      u8::from_str_radix(
        line.next()
        .with_context(||format!("ising_io.rs: Not enough entries for hex code \
            of {state_type} config"))?,16
        ).with_context(||format!("ising_io.rs: {state_type} not given in hex?")
        )?);
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
      Ok(Some(usize_to_config(config_usize, ising_instance.n_points())))
      } 
    }
  }
}
pub fn from_ising_file<I:Ising>(path: impl AsRef<Path>) -> 
Result<(I, f64, Option<BitVec>, Option<BitVec>)>
{
  let file = File::open(path).unwrap();
  let mut reader = BufReader::new(file);
  ensure!(
    find_section(&mut reader, "$temp")?,
    "ising_io.rs: could not find temperature"
  );
  let temp:f64 = read_single_entry(&mut reader, "temperature")?;
  ensure!(
    find_section(&mut reader, "$rng_seed")?,
    "ising_io.rs: could not find rng_seed"
  );
  let seed_string: String= read_single_entry(&mut reader, "seed")?;
  let t = seed_string.trim();
  let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X"))
    .unwrap_or(t);
  let seed = u64::from_str_radix(t, 16)
    .with_context(|| format!("bad seed {seed_string:?}, expected \
        hexadecimal"))?;
  ensure!(
    find_section(&mut reader, "$dim_sizes")?,
    "ising_io.rs: could not find dimensions and sizes"
  );
  let dim_sizes_vec:Vec<usize> = 
    parse_line(&mut reader, "dimensions and sizes")?;
  let (&dim, sizes) = dim_sizes_vec
    .split_first()
    .context("ising_io.rs: dimensions line is empty")?;
  ensure!(dim > 0, "ising_io.rs: dimension cannot be 0");
  ensure!(
    sizes.len() == dim,
    "ising_io.rs: we are given {dim} dimensions but {} sizes were given",
    sizes.len()
  );
  ensure!(sizes.iter().all(|&s| s>0), "all sizes must be nonzero: {sizes:?}");
  let sizes = sizes.to_vec();
  ensure!(
    find_section(&mut reader, "$mu")?,
    "ising_io.rs: could not find mu"
  );
  let mu:f64 = read_single_entry(&mut reader, "mu")?;

  ensure!(
    find_section(&mut reader, "$external_magnetic_field")?,
    "ising_io.rs: could not find external_magnetic_field"
  );
  let n_points:usize = sizes.iter().product();
  let external_magnetic_field:Vec<f64>
    = (0..n_points).map(|i|
      read_single_entry(&mut reader, "external magnetic field"
        ).with_context(|| format!("ising_io.rs: external magnetic field: \
          expected {n_points} entries, failed at entry {i}"))
      ).collect::<Result<_>>()?;
  let mut ising_instance = I::new(sizes, seed, external_magnetic_field, mu,
    dim * 2);
  ensure!(
    find_section(&mut reader, "$edge_weights_start")?,
    "ising_io.rs: could not find edge weights"
  );
  let weights = read_weights(&mut reader, &ising_instance)?;
  ising_instance.set_edges(weights);

  let in_section = find_section(&mut reader, "$starting_configs").unwrap();
  let starting_configs: Option<Vec<StartingConfig>>;
  if in_section {
    starting_configs = read_starting_configs(&mut reader, &ising_instance)?;
  }
  else {
    #[cfg(debug_assertions)]
    {
    eprintln!("ising_io.rs warning: Starting Configurations not found. \
      Defaulting to random Starting Configurations");
    }
    starting_configs = None;
  }
  ising_instance.set_starting_configs(starting_configs);
  
  let in_section = find_section(&mut reader, "$ground_state_approx").unwrap();
  
  let ground_state_approx:Option<BitVec> = if !in_section{
    None
  } else {
    read_state(&mut reader, &ising_instance, "ground_state")?
  };
  
  let in_section = find_section(&mut reader, "$local_min_approx").unwrap();
  let loc_min_approx:Option<BitVec> = if !in_section{
    None
  } else {
    read_state(&mut reader, &ising_instance, "local_min_approx")?
  };

  ising_instance.init_spins();
  ising_instance.init_cost();
  Ok((ising_instance, temp, ground_state_approx, loc_min_approx))
}

fn usize_to_config(uvec:Vec<usize>, n_points:usize)->BitVec{
    let mut bit_vec = BitVec::from_vec(uvec);
    bit_vec.truncate(n_points);
    bit_vec
}

pub fn append_section(
  path: impl AsRef<Path>,
  section_header: &str,
  section_value: &str,
  ) -> Result<()>{
  let path = path.as_ref();
  let write_err = || "ising_io.rs error: could not write to input file";
  let file = File::open(path)
    .context("ising_io.rs: could not open input file")?;
  
  let mut reader = BufReader::new(file);
  if find_section(&mut reader,section_header).context("could not \
    read through file")?{
    anyhow::bail!(
      "ising_io.rs error: {section_header} already exists. \
      remove this section from the input file before running this code"
    );
  };
  drop(reader);
  let mut file = OpenOptions::new()
    .append(true)
    .open(path)
    .context("ising_io.rs: could not open input file for appending")?;

  writeln!(file).with_context(write_err)?;
  writeln!(file, "{section_header}").with_context(write_err)?;
  writeln!(file, "{section_value}").with_context(write_err)?;

  file.flush().context("ising_io.rs: could not flush input file")?;
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

pub fn finite_temperature_option<I:Ising>(
  ising_model: &mut I,
  max_steps: usize,
  runs: usize,
  stationary_file_name: PathBuf,
  temperature:f64,
  ground_state_opt:Option<BitVec>,
  ground_state_provided:bool,
  ) -> Result<()>{
  let write_err = || format!("ising_io.rs: error writing to stationary file \
    {}", stationary_file_name.display());
  let stationary_file = File::create(&stationary_file_name).with_context
    (|| format!("ising_io.rs error:could not create new stationary file {}",
                stationary_file_name.display()))?;
  let mut stationary_out = BufWriter::new(stationary_file);
  if ground_state_provided{
    if ground_state_opt.is_none(){
      bail!("ising_io.rs: no ground state provided");
    }
    let ground_state = ground_state_opt.unwrap();
    for _ in 0..runs{
    let hitting_time = stationary_finite_temperature_hit(
      ising_model,
      &ground_state,
      max_steps,
      temperature
    );
      match hitting_time{
        Some(steps) => writeln!(stationary_out, "{}", steps)
          .with_context(write_err)?,
        None => writeln!(stationary_out, "NA").with_context(write_err)?
      }
      stationary_out.flush().with_context(write_err)?;
      ising_model.reinitialize();
    } 
  } else {
    writeln!(stationary_out, "{}\t{}\t{}\t{}", "best_configuration",
      "best_cost", "best_cost_fpb", "hitting_time").with_context(write_err)?;
    for _ in 0..runs{
      let (best_conf, best_cost, hit) 
        = stationary_finite_temperature_optimise(
          ising_model,
          max_steps,
          temperature);
      let best_conf_string = bitvec_to_hex_string(&best_conf);
      writeln!(stationary_out, "{}\t{:.6e}\t{:016x}\t{}", best_conf_string,
        best_cost, best_cost.to_bits(), hit).with_context(write_err)?;
      ising_model.reinitialize();
      stationary_out.flush().with_context(write_err)?;
    }
  }
  Ok(())
}

pub fn infinite_temperature_option<I:Ising>(
  ising_model: &mut I,
  max_steps:usize,
  runs:usize,
  stationary_file_name:PathBuf,
  ground_state_opt:Option<BitVec>,
  ground_state_provided:bool
  ) -> Result<()>{
  let write_err = || format!("ising_io.rs: error writing to stationary file \
    {}", stationary_file_name.display());
  let stationary_file = File::create(&stationary_file_name)
    .with_context(||format!("ising_io.rs: could not create stationary file {}", 
               stationary_file_name.display()))?;
  let mut stationary_out = BufWriter::new(stationary_file);
  if ground_state_provided{
    if ground_state_opt.is_none(){
      bail!("ising_io.rs: no ground state was provided. Exiting out.");
    }
    let ground_state = ground_state_opt.unwrap();
    for _ in 0..runs{
      let hitting_time = stationary_infinite_temperature_hit(
        ising_model,
        &ground_state,
        max_steps
        );
      match hitting_time{
        Some(steps) => writeln!(stationary_out, "{}", steps)
          .with_context(write_err)?,
        None => writeln!(stationary_out, "NA").with_context(write_err)?
      }
      stationary_out.flush().with_context(write_err)?;
      ising_model.reinitialize();
    }
  } else {
    writeln!(stationary_out, "{}\t{}\t{}\t{}", "best_configuration",
      "best_cost", "best_cost_fpb", "hitting_time")
      .with_context(write_err)?;
    for _ in 0..runs{
      let (best_conf, best_cost, hit)
        = stationary_infinite_temperature_optimise(
          ising_model,
          max_steps);
      let best_conf_string = bitvec_to_hex_string(&best_conf);
      writeln!(stationary_out, "{}\t{:.6e}\t{:016x}\t{}", best_conf_string,
        best_cost, best_cost.to_bits(), hit).with_context(write_err)?;
      stationary_out.flush().with_context(write_err)?;
      ising_model.reinitialize();
    }
  }
  Ok(())
}

pub fn high_conv_output<I:Ising>(temperature:f64, human_readable:bool,
  ground_state_opt:Option<&BitVec>, ising_model:&mut I) -> (bool, f64){
  let (sign, log_sum) = high_temp_theoretical_perturbation_naive(
    ising_model,
    ground_state_opt,
    temperature);
    if human_readable{
        println!("log2 absolute perturbation:{}", log_sum);
        println!("sign:{}", if sign {"-"} else {"+"});
      } else {
        println!("{}, {}", log_sum, if sign {"-"} else {"+"});
      }
    (sign, log_sum)
}

pub fn low_conv_output<I:Ising>(
  temperature:f64,
  human_readable:bool,
  local_min_approx_opt:Option<&BitVec>,
  ising_model: &mut I) -> Result<f64>{
  let local_min_approx = local_min_approx_opt.context(LOCAL_MIN_MISSING_ERR)?;
  let perturbation = ising_model.low_temp_pert(local_min_approx, temperature);
  if human_readable{
    println!("low temperature perturbation: {}", perturbation)
  }
  else {
    println!("{}", perturbation);
  }
  Ok(perturbation)
}

pub fn asymptotic_match_output<I:Ising>(
  temperature:f64,
  human_readable:bool,
  local_min_approx_opt:Option<&BitVec>,
  ground_state_approx_opt:Option<&BitVec>,
  ising_model:&mut I) -> Result<f64>{
  let local_min_approx = local_min_approx_opt.context(LOCAL_MIN_MISSING_ERR)?;
  let ground_state_approx = ground_state_approx_opt
    .context(GLOBAL_MIN_MISSING_ERR)?;
  let perturbation = asymptotic_matching(ising_model,
    local_min_approx, ground_state_approx, temperature);
  if human_readable{
    println!("asymptotic match: {}", perturbation);
  } else {
    println!("{}", perturbation);
  }
  Ok(perturbation)
}

pub fn compute_hitting_time_output_high_t<I:Ising>(
  human_readable:bool,
  percent_opt:Option<f64>,
  perturbation_result:(bool, f64),
  ising_model: &mut I) -> Result<()> {
  let percent = percent_opt.unwrap_or(0.25);
  let steps = approx_num_steps_for_percent_log_perturbation
    (ising_model, percent, None, Some(perturbation_result), None)?;
  if human_readable {
    println!("Takes ~ {steps} steps to have {percent} \
              probability of hitting")
  } else {
    println!("{}, {}", steps, percent);
  }
  Ok(())
}

pub fn compute_hitting_time_output_low_mid_t<I:Ising>(
  human_readable:bool,
  percent_opt:Option<f64>,
  perturbation:f64,
  ising_model: &mut I) -> Result<()>{
  let percent = percent_opt.unwrap_or(0.25);
  let steps = approx_num_steps_for_percent(ising_model, percent,
    None, Some(perturbation), None)?;
  if human_readable {
    println!("Takes ~ {steps} steps to have {percent} \
      probability of hitting");
  } else { println!("{}, {}", steps, percent); }
  Ok(())
}

pub fn greedy_approx_output<I:Ising>(
  inner_sweeps:usize,
  outer_sweeps:usize,
  human_readable:bool,
  loc_min_approx:bool,
  modify_input_file:bool,
  loc_min_retries:usize,
  loc_min_recursive:bool,
  ising_model:&mut I,
  input_file:PathBuf) -> Result<()>{
  if human_readable && modify_input_file {
    bail!("ising_io.rs: greedy_approx: human_readable and \
      modify_input_files cannot both be set.");
  }
  match loc_min_approx {
    false =>{
  let (ground_state_approx, _)=greedy(
    ising_model,
    inner_sweeps,
    outer_sweeps);

  if human_readable {
    println!("Greedy algorithm found the following configuration:");
    println!("{}",
      ground_state_approx.iter()
        .by_vals()
        .map(|spin| if !spin {"-1"} else {"+1"})
        .collect::<Vec<_>>()
        .join(" ")
    );
    return Ok(());
  }
  let config_str: String = bitvec_to_hex_string(&ground_state_approx);
  if !modify_input_file{
    println!("{}", config_str);
   }
  if modify_input_file{
    append_section(
      input_file, "$ground_state_approx", &config_str
      )?;
    }
  },
  true =>{
    let (global_state_approx, local_state_approx) = 
      local_global_min_approx(ising_model, outer_sweeps, inner_sweeps, 
        loc_min_retries, loc_min_recursive);
    if human_readable {
    println!("Greedy algorithm found the following configuration for \
      ground state:");
    println!("{}",
      global_state_approx.iter()
        .by_vals()
        .map(|spin| if !spin {"-1"} else {"+1"})
        .collect::<Vec<_>>()
        .join(" ")
    );
    println!("Greedy algorithm found the following configuration for \
      local minimum:");
    println!("{}",
      local_state_approx.iter()
        .by_vals()
        .map(|spin| if !spin {"-1"} else {"+1"})
        .collect::<Vec<_>>()
        .join(" ")
    );
    return Ok(());
  }
  let ground_string: String = bitvec_to_hex_string(&global_state_approx);
  let local_string: String = bitvec_to_hex_string(&local_state_approx);
  
  if !modify_input_file{
    println!("{}", ground_string);
    println!("{}", local_string);
  }
  if modify_input_file{
    append_section(
      &input_file, "$ground_state_approx", &ground_string
    )?;
    append_section(
      &input_file, "$local_min_approx", &local_string
    )?;
      }
    }
  }
  Ok(())
}
