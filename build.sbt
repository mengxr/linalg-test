name := "linalg-test"

version := "0.1"

libraryDependencies += "org.scalanlp" % "breeze_2.10" % "0.7"

libraryDependencies += "org.apache.mahout" % "mahout-math" % "0.9"

libraryDependencies += "org.jblas" % "jblas" % "1.2.3"

libraryDependencies += "com.googlecode.matrix-toolkits-java" % "mtj" % "1.0.1" // exclude("com.github.fommil.netlib", "all")

libraryDependencies += "org.apache.commons" % "commons-math3" % "3.2"

libraryDependencies += "edu.berkeley.bid" %% "bidmat" % "0.1.0"

libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2"

libraryDependencies += "log4j" % "log4j" % "1.2.17"

net.virtualvoid.sbt.graph.Plugin.graphSettings
