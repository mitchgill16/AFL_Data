###
#! /usr/bin/Rscript

library('fitzRoy')
library('dplyr')

args = commandArgs(trailingOnly=TRUE)
getwd()
year <- as.integer(args[1])
rnd <- as.integer(args[2])

#make Pavs for given year