# Project Specification

## Overview

A Rust program that evolves AI combatants for a 1v1 spaceship duel game, inspired by the original Atari Asteroids arcade game. There are no asteroids — just two ships trying to destroy each other. Ships are controlled by genome-encoded algorithms that improve over generations through genetic reinforcement learning.

## Requirements

### Game Mechanics
- Two ships in a 1000x1000 2D arena with toroidal wrapping (objects leaving one edge reappear on the opposite side, just like the original Asteroids)
- Each ship is a triangle, visually similar to the player ship in Asteroids
- Thrust-based movement: ships have rotation, forward thrust, and inertia (no instant direction changes)
- Each ship can fire projectiles; bullets travel a maximum of 200px before disappearing
- Each ship gets 8 bullets per round (once fired, bullets are gone until the next round)
- A ship is destroyed when hit by an opponent's projectile
- A match ends when one ship is destroyed or a time limit is reached

### Evolutionary System
- Each ship's behavior is controlled by an algorithm described by its genome
- The structure of the genome and how it maps to ship behavior is an open design decision
- Use some form of genetic reinforcement learning to select for better competitors over generations
- The system should run many matches per generation to evaluate fitness
- Evolution should produce increasingly competent fighters over time

### Visualization
- While evolution is running, the user must be able to watch matches being played out in real time
- The visualization should show the current state-of-the-art evolved players competing
- Rendering should evoke the original Asteroids aesthetic (line graphics on dark background)

## Constraints

- Written in Rust
- Must compile and run on Linux
