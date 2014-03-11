package dense.add.sparse

import util.Benchmark
import java.util.Random
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{Vector => BV, DenseVector => BDV, VectorBuilder => BVB, sum => breezeSum}
import org.apache.mahout.math.{DenseVector => MDV, SequentialAccessSparseVector => MSV}

abstract class DenseAddSparseBenchmark extends Benchmark {
  val n = 1000000
  val random = new Random(0)
  val arr = Array.fill(n)(random.nextDouble())
  val sparsity = 0.02
  val indicesBuffer = ArrayBuffer[Int]()
  val valuesBuffer = ArrayBuffer[Double]()
  (0 until n).foreach { i =>
    val x = random.nextDouble()
    if (x < sparsity) {
      indicesBuffer += i
      valuesBuffer += x
    }
  }
  val indices = indicesBuffer.toArray
  val values = valuesBuffer.toArray
}

class NaiveDenseAddSparseBenchmark extends DenseAddSparseBenchmark {

  def run() {
    var i = 0
    val nnz = indices.length
    while (i < nnz) {
      arr(indices(i)) += values(i)
      i += 1
    }
  }

  def certificate(): Double = arr.sum
}

class BreezeDenseAddSparseBenchmark extends DenseAddSparseBenchmark {

  val dv: BV[Double] = new BDV[Double](arr)

  val svBuilder = new BVB[Double](n)
  indices.zip(values).foreach { case (i, x) =>
    svBuilder.add(i, x)
  }
  val sv: BV[Double] = svBuilder.toSparseVector

  def run() {
    dv += sv
  }

  def certificate(): Double = breezeSum(dv)
}

class MahoutDenseAddSparseBenchmark extends DenseAddSparseBenchmark {

  val dv = new MDV(arr)

  val sv = new MSV(n)
  indices.zip(values).foreach { case (i, x) =>
    sv.set(i, x)
  }

  def run() {
    dv.addAll(sv)
  }

  def certificate(): Double = dv.zSum()
}

object DenseAddSparseBenchmarks extends App {
  val m = 10000
  val numTrials = 1000
  val naive = new NaiveDenseAddSparseBenchmark
  val breeze = new BreezeDenseAddSparseBenchmark
  val mahout = new MahoutDenseAddSparseBenchmark
  for (bench <- Seq(naive, breeze, mahout)) {
    bench.runBenchmark(m, numTrials)
  }
}