package dense.plus.sparse

import java.util.Random

import util.Benchmark
import breeze.linalg.{DenseVector => BreezeDenseVector, Vector => BreezeVector, VectorBuilder}
import org.apache.mahout.math.{SequentialAccessSparseVector, DenseVector => MahoutDenseVector, Vector => MahoutVector}
import org.apache.mahout.math.function.Functions
import no.uib.cipr.matrix.sparse.{SparseVector => MtjSparseVector, CompRowMatrix => MtjSparseMatrix}
import no.uib.cipr.matrix.{DenseVector => MtjDenseVector, Vector => MtjVector}

abstract class DensePlusSparseBenchmark extends Benchmark {

  val n = 1000000
  val arr = Array.fill(n)(0.0)
  val sparsity = 0.05
  val random = new Random(0)
  val elements = (0 until n).filter( x => random.nextDouble() < sparsity )
    .map((_, random.nextDouble()))
}

class BreezeDensePlusSparseBenchmark extends DensePlusSparseBenchmark {

  val d: BreezeDenseVector[Double] = new BreezeDenseVector[Double](arr)
  val sb = new VectorBuilder[Double](n, elements.length)
  elements.foreach { e =>
    sb.add(e._1, e._2)
  }
  val s: BreezeVector[Double] = sb.toSparseVector

  override def run() {
    d += s
  }

  override def certificate(): Double = d(elements.head._1)
}

class MahoutDensePlusSparseBenchmark extends DensePlusSparseBenchmark {

  val d: MahoutVector = new MahoutDenseVector(arr)
  val s: MahoutVector = new SequentialAccessSparseVector(n, elements.size)
  elements.foreach { e =>
    s.set(e._1, e._2)
  }

  override def run() {
    d.assign(s, Functions.PLUS)
  }

  override def certificate(): Double = d.get(elements.head._1)
}

class MtjDensePlusSparseBenchmark extends DensePlusSparseBenchmark {

  val d: MtjDenseVector = new MtjDenseVector(arr)
  val indices = new Array[Int](elements.length)
  val values = new Array[Double](elements.length)
  var i = 0
  elements.foreach { e =>
    indices.update(i, e._1)
    values.update(i, e._2)
    i += 1
  }
  val s: MtjVector = new MtjSparseVector(n, indices, values)

  override def run() {
    d.add(s)
  }

  override def certificate(): Double = d.get(0)
}

class NaiveDensePlusSparseBenchmark extends DensePlusSparseBenchmark {

  val d = arr.clone()
  val indices = new Array[Int](elements.length)
  val values = new Array[Double](elements.length)
  var i = 0
  elements.foreach { e =>
    indices.update(i, e._1)
    values.update(i, e._2)
    i += 1
  }

  override def run() {
    var i = 0
    var index = 0
    while (i < indices.length) {
      index = indices(i)
      d.update(index, d(index) + values(i))
      i += 1
    }
  }

  override def certificate(): Double = d(elements.head._1)
}


object DensePlusSparseBenchmarks extends App {

  val m = 10000
  val numTrials = 1000

  val mahout = new MahoutDensePlusSparseBenchmark()
  mahout.runBenchmark(m, numTrials)
  val breeze = new BreezeDensePlusSparseBenchmark()
  breeze.runBenchmark(m, numTrials)
  val naive = new NaiveDensePlusSparseBenchmark()
  naive.runBenchmark(m, numTrials)
}