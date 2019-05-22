# What?

Sample project directory and structure for deep learning projects in pytorch.

# Why? 

Everyone has their own way of doing things. That's fine. Whatever floats your boat. But over the course of my deep learning shenanigans I've found a few things. First, that pytorch is much more friendly to use than tensorflow. Second, this is the folder structure and ontological organization that makes the most sense, at least to me.

So, I made this because I'm sick of adapting the crappy code I find on Github. This repo is my argument for how things ought to be structured.

Use it if you want. Or don't. Whatever.

# How?

If you find this compelling enough to adopt for your own, simply just run

```
git clone https://github.com/handrew/torch-project.git
cd torch-project
rm -rf .git
```

Then, to get started, you simply modify the requirements.txt, .gitignore according to your needs, install dependencies, write download scripts, fill out relevant model definintions in `models/`, preprocessing scripts in `utils`, and write your training function in `main`. 
