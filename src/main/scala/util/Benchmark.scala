package util

trait Benchmark {

   def run(): Unit

   def certificate(): Double

   def runBenchmark(n: Int, numTrials: Int = 0) {

     // run once and get certificate
     this.run()
     println(this.getClass.getName + " certificate: " + this.certificate())

     // warm up
     for(i <- 1 to numTrials) {
       this.run()
     }

     // timing
     val start = System.nanoTime()
     for(i <- 1 to n) {
       this.run()
     }
     val duration = System.nanoTime() - start
     println(this.getClass.getName + " time: " + duration / 1e6 / n + "ms")
   }
 }
