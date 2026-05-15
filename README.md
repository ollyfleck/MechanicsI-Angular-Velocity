# MechanicsI-Angular-Velocity

This application is a learning experience about agentic programming - 99% of the code in this repository has been written by various LLMs (and it shows) that I deployed locally on my own hardware. 
For the majority of the programming I used qwen3.6-35b-a3b due to me having a graphics card with a relatively low amount of VRAM. 

What I have found out (at least when running small local models):
- Agentic programming is clearly not there yet. It absolutely needs a human to oversee what it does, and plan out a path. Agents are very helpful for analyzing code, and autocompleting functions, but hands off 100% autonomous programming leads to the code quickly becoming a huge mess.
- The quality of the code is just... mid, and it tends to be significantly more scattered and duplicated than code in repositories made by humans, even if they have thousands of contributors.
- Models struggle with keeping large codebases in context, while some models support up to 1M tokens of context, even that usually isn't enough, and larger context windows lead to loss of coherence, so there is an element of diminishing returns.
- The cost of locally deployed models is relatively low compared to frontier models, but their capabilities are significantly reduced. Frontier models hover around trillions of parameters, which is orders of magnitude above what most consumer hardware can comfortably deploy.

A frontier model specifically designed for agentic coding such as Anthropic's Opus models will give you much higher quality code, and generate it faster, but obviously, that's going to cost a lot more.
At some point your bills to frontier model provides with exceed the monthly salary of a senior software engineer, and your codebase will continue to get worse and even less maintainable. Much like a drug addiction, quitting is harder the longer you use, and it ends up ruining your finances.
