package weka.classifiers.mmall.Ensemble.objectiveFunction.parallel;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

//import lbfgsb.FunctionValues;
import weka.classifiers.mmall.optimize.FunctionValues;

import weka.classifiers.mmall.Ensemble.objectiveFunction.ObjectiveFunctionCLL_w;
import weka.classifiers.mmall.Ensemble.wdAnJE;
import weka.classifiers.mmall.Utils.SUtils;

public class ParallelObjectiveFunctionCLL_w extends ObjectiveFunctionCLL_w {

	int nThreads;
	double[][] gs;
	private double[][] tmpProbs;
	private ExecutorService executor;
	int N;
	
	private static final int minNPerThread = SUtils.minNumThreads;

	public ParallelObjectiveFunctionCLL_w(wdAnJE algorithm) {
		super(algorithm);
		this.N = super.algorithm.getNInstances();
		
		if (N < minNPerThread) {
			this.nThreads=1;
		} else {
			this.nThreads = Runtime.getRuntime().availableProcessors();
			if (N/this.nThreads < minNPerThread) {
				this.nThreads = N/minNPerThread + 1;
			}
		}
		System.out.println("launching "+nThreads+" threads");

		this.gs = new double[nThreads][super.algorithm.getdParameters_().getNumberParametersAllocated()];
		this.tmpProbs = new double[nThreads][super.algorithm.getNc()];
		executor = Executors.newFixedThreadPool(nThreads);
	}

	@Override
	public FunctionValues getValues(double params[]) {

		double negLogLikelihood = 0.0;
		algorithm.getdParameters_().copyParameters(params);
		double g[] = new double[algorithm.getdParameters_().getNumberParametersAllocated()];

		Future<Double>[] futures = new Future[nThreads];

		int assigned = 0;
		int remaining = algorithm.getNInstances();
		
		for (int th = 0; th < nThreads; th++) {
			/*
			 * Compute the start and stop indexes for thread th
			 */
			int start = assigned;
			int nInstances4Thread = remaining / (nThreads - th);
			assigned += nInstances4Thread;
			int stop = assigned - 1;
			remaining -= nInstances4Thread;

			/*
			 * Calling thread
			 */
			//Callable<Double> thread = new CallableCLL_w(algorithm.getM_Instances(), start, stop, algorithm.getNc(), algorithm.getnAttributes(), algorithm.getMS(), tmpProbs[th], gs[th], algorithm.getdParameters_());
			//System.out.println("Starting at: " + start + ", finishing at: " + stop);
			Callable<Double> thread = new CallableCLL_w(start, stop, tmpProbs[th], gs[th], algorithm, th);
			
			futures[th] = executor.submit(thread);
		}
		
		for (int th = 0; th < nThreads; th++) {
			try {
				negLogLikelihood += futures[th].get();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			for (int i = 0; i < g.length; i++) {
				g[i] += gs[th][i];
			}
		}

		return new FunctionValues(negLogLikelihood, g);
	}

	@Override
	public void finish(){
		executor.shutdown();
	}

}
