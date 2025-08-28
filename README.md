# Dep La: An AI for Fantasy Premier League
Dep La is an AI system built to predict [Fantasy Premier League](https://fantasy.premierleague.com/) points, and optimize Fantasy Premier League teams. This project uses machine learning algorithms to analyze past data and predict future player and team performances. 

> [!NOTE]
> Dep La is live for the 2025/26 season. Keep track of its performance [here](https://fantasy.premierleague.com/entry/71095/history). 

## Approach
This is an outline of the approach Dep La uses to optimize FPL teams: 

1. **Collect data**: The first step is to collect historical and upcoming data for both players and teams. This data is collected from multiple sources and stored in [this repository](https://github.com/kz4killua/fpl-data). 
2. **Engineer features**: Using the raw data, we extract several machine learning features. Which players are in great form? How difficult are the upcoming games? How well is each team playing? How well does each player perform at home vs away? 
3. **Make predictions**: The next step is to predict how many points each player will score in their next matches. This is done using a mixture of machine learning (e.g. regression) and statistical techniques. 
4. **Optimize fantasy teams**: Predicting points is only half the story. We also need to figure out which fantasy players to buy, which ones to sell, who to captain, who to start, who to bench, etc. Currently, this is done using a [mixed integer linear programming](https://en.m.wikipedia.org/wiki/Integer_programming) algorithm. 

## Running Locally
_Coming soon..._

## Credits
The following resources were helpful when writing this program:
- _Coming soon..._
