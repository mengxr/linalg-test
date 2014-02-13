package util

trait Benchmark {

   def run(): Unit

   def runBenchmark(n: Int, numTrials: Int = 0) {
     for(i <- 1 to numTrials) {
       this.run()
     }
     val start = System.nanoTime()
     for(i <- 1 to n) {
       this.run()
     }
     val duration = System.nanoTime() - start
     println(this.getClass.getName + ": " + duration / 1e6 / n + "ms")
   }
 }
