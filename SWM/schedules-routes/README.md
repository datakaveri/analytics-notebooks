# SWM Varanasi Route and Schedule Analysis

## The notebook contains:

- Data Exploration and Cleaning of the Varanasi SWM data.
- Identification of the static routes and schedules.

## Data Exploration
The data is preprocessed and the vehicles that do not have enough data for any significant analysis have been cleaned out. The area covered by the vehicles, the dustbins’ locations and some daily trends have been explored and visualised.

## Identification of Routes
Routes for various different kinds of vehicles have been analysed and some consistency in the routes has been found. Routes are determined by using an algorithm that uses map matching techniques.

## Identification of Schedules
The general schedules are computed using an algorithm involving mapping and averaging techniques to minimise errors. The H3 library is used to bucket points together.
