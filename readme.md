# Linux Process Scanning

I can make a base model with the NSW15 datasets, but I want to have a model which trains off of the user's behavior to see anything out of the ordinary

We need to have a way to use both, the base model with be more trustworthy, and the extended model can catch things but will be more likely to gove out false positives (probably, I should look into this)

SO we need to balance them

## Balancing

I think there should be an algorithm, weight the base model as 70% of the determination, and the extended as 30% or somehthing like that

If the extended is very sure about something it should be flagged, and if the base model is adamant that it's not malicious, it's probably fine

## Feedback Loop

Things only get better with feedback

If the base model says something is benign and the extended model says it's malicious then something is wrong, either the extended model should be tuned or the base model should be updated

## Dataset

Righ now using features which I can read with python and are in the dataset and with a few different models I'm getting an 88% accuracy. There are other features of processes which aren't included in the dataset (I'll have to read the paper to find out why) but I think would make the detection rate much better, things like 

- Memory usage: p.memory_info().rss, vms, etc
- Thread info: number of threads (p.num_threads()), thread states if available
- I/O behavior: p.io_counters() – bytes read/written, syscall counts
- Open file count: len(p.open_files()) – many malware touch files rapidly
- Network activity: number of open sockets (len(p.connections()))
- Context switches: p.num_ctx_switches() (voluntary/involuntary)

This means that I would have to make a new dataset. For now I'll just use the existing, but I want to also look into making my own in the future, maybe with cyber club
