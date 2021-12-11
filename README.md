# Procedurally Generated Mountain Village 

![procgen-terrain](example_images/village.gif)

## Setting up Vulkan

The code works both on Linux and Windows. For Linux the script `setup_vulkan_deps.sh` should do most of the Vulkan setup required as long as your graphics driver is installed. 

On Windows additional setup for Vulkan is not necessary.

### Setting up Rust 

The recommended way to setup Rust is with [rustup.sh](https://www.rust-lang.org/learn/get-started)

Install the stable track of Rust 

```sh
rustup install stable
```

### Additional Dependencies

You will also need to install `Python`, `CMake`, and `ninja-build` to compile the shaders. Install those via your preferred method for your platform.


### Building the application 

Change into the `procgen-terrain` directory.

You can build the app using 

```sh
cargo build --release
```

The application will be found in `target/release/procgen-terrain`

You can just run the application to use the default settings. However it can take a number of command line arguments to control the generation process. 

```
procgen-terrain 0.1.0

USAGE:
    procgen-terrain.exe [OPTIONS]

FLAGS:
        --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -b, --bounds <bounds>         [default: 30]
    -g, --gridsize <gridsize>     [default: 256]
    -h, --houses <houses>         [default: 10]
    -m, --map <map>               [default: 128.0]
    -n, --noise <noise>           [default: 512]
    -s, --seed <seed>             [default: 100]
    -t, --trees <trees>           [default: 200]
```

### Options:

- `noise`: Control the size of the noise map (NxN) that is generated. This allows you to control the resolution of the noise 
- `map`: Size (MxM) of the terrain mesh generated in world coordinates
- `gridsize`: The resolution (GxG) of the terrain mesh. Higher numbers mean more vertices in the mesh
- `seed`: The seed for the noise generator. Giving the same seed generates the same noise map
- `bounds`: The distances from the center that modeled geometry can spawn. Prevents unnecessary rendering for models that can't be seen 
- `trees`: Number of trees to spawn. Trees are spawned in random positions on the terrain within the bounds 
- `houses`: Number of houses to spawn. Like trees houses are spawned in random positions within the bounds


### Cool Seeds:

```
200120
42
100
1337
238121
```