package adventofcode.solutions

import adventofcode.Definitions.*

@main def Day01 = Day(1) { (input, part) =>

  val sums = input.split(lineSeparator * 2).map(_.split(lineSeparator).map(_.toInt).sum)
  

  println(input.split(lineSeparator)+"")//.foreach(x => println(x))


  part(1) = 0

  part(2) = 0

}