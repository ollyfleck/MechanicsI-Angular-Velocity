# MechanicsI-Angular-Velocity

This readme was written entirely by hand. This application is a learning experience about agentic programming - 99% of the code in this repository has been written by various LLMs (and it shows) that I deployed locally on my own hardware. 
For the majority of the programming I used qwen3.6-35b-a3b due to me having a graphics card with a relatively low amount of VRAM. 
I used Cline in VSCode for my agentic programming harness.

What I have found out (at least when running small local models):
- Agentic programming is clearly not there yet. It absolutely needs a human to oversee what it does, and plan out a path. Agents are very helpful for analyzing code, and autocompleting functions, but hands off 100% autonomous programming leads to the code quickly becoming a huge mess.
- The quality of the code is just... mid, and it tends to be significantly more scattered and duplicated than code in repositories made by humans, even if they have thousands of contributors.
- Models struggle with keeping large codebases in context, while some models support up to 1M tokens of context, even that usually isn't enough, and larger context windows lead to loss of coherence, so there is an element of diminishing returns.
- The cost of locally deployed models is relatively low compared to frontier models, but their capabilities are significantly reduced. Frontier models hover around trillions of parameters, which is orders of magnitude above what most consumer hardware can comfortably deploy.

A frontier model specifically designed for agentic coding such as Anthropic's Opus models will give you much higher quality code, and generate it faster, but obviously, that's going to cost a lot more.
At some point your bills to frontier model provides with exceed the monthly salary of a senior software engineer, and your codebase will continue to get worse and even less maintainable. Much like a drug addiction, quitting is harder the longer you use, and it ends up ruining your finances.

# Cost analysis
Some rough calculations for the power usage of this ordeal:
At idle, my entire computer setup draws about 190W, this jumps up to 250W during prompt processing and 290W during token generation. 
Let's assume the worst case scenario and say that if the model is actively working, it's consuming the full 290W (even though my computer is still usable as the model doesn't take up 100% of its resources)
Over the course of a month I let the agent work on the project on and off. This was split between prompting, waiting for prompt processing + token generation, analyzing outputs, and just kind of doing nothing sometimes because I didn't realise the harness asked me to approve a change to a file or run a command.
Let's assume the worst again, by saying that the model worked for a full 8 hours a day for an entire month, which is a rather large overestimation.
We have power consumption, and a time span. We can figure out the energy:

8 hours * 31 days = 248 hours

290W * 248 hours = 71,920 Wh ≈ 72kWh

This is roughly equivalent to a standard energy-efficient LED lightbulb running 24 hours per day for an entire year, or 8 hours a day for 3 years. 
It is also equivalant to using an electric kettle to make a cup of tea almost every single day for a whole year. 

The electricity cost for an average residential household in Poland as of the time I'm writing is about 1.05 PLN/kWh
This means that if I did actually use the model to work on the application for 8 hours a day every single day, it would run me an additional 75PLN in utility bills. Realistically though, I'd say it's at least half that, maybe even a third. 

As a comparison, here's the price of this project if I had used a frontier provider:
A Claude code subsciption is $17 a month (if billed annually) which as of the time of writing is about 63PLN, and this doesn't include billing for actually using a model. The pricing system is complicated as it includes different prices for new and cached tokens AND cache hits/misses, but a project like this would cost anywhere from $5 to something like $20, based on the number of tokens I had to generate in total, which I estimate to be around 4-8 million.
