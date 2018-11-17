package weka.classifiers.mmall.Online.Bayes;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;

import hdp.ProbabilityTree;
import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.Utils.CorrelationMeasures;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;

public final class BNStructure {

	private int[][] m_Parents;
	private int[] m_Order;

	int nInstances;
	int nAttributes;
	int nc;
	xxyDist xxyDist_;
	private String m_S = "";
	private int K = 1;
	private long maxNFreeParams = Long.MAX_VALUE;

	public ArrayList<int[]> upperOrder;
	public ArrayList<int[][][]> parentOrder;

	protected static int MAX_INCORE_N_INSTANCES = 100000;

	public int ensembleSize = 5;
	ArrayList<ArrayList<ArrayList<Integer>>> parentOrderforEachAtt;

	private int[] paramsPerAtt;
	private int m_BestK_ = 0;
	private int m_BestattIt = 0;
	ArrayList<HashMap<ArrayList<Integer>,Integer>> parentForEachAtt2;

	public BNStructure(Instances m_Instances, String m_S, int k, int[] ppa) {
		this.m_S = m_S;
		this.K = k;

		nInstances = m_Instances.numInstances();
		nAttributes = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();

		m_Parents = new int[nAttributes][];
		m_Order = new int[nAttributes];

		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = ppa[u];
		}

		parentOrder = new ArrayList<int[][][]>();
		upperOrder = new ArrayList<int[]>();

		for (int i = 0; i < nAttributes; i++) {
			m_Order[i] = i;
		}

		xxyDist_ = new xxyDist(m_Instances);
		if (nInstances > 0) {
			xxyDist_.addToCount(m_Instances);
		}
	}

	public void learnStructure(Instances structure, File sourceFile, RandomGenerator rg) throws IOException {

		// First fill xxYDist; everybody needs it
		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		Instance row;

		while ((row = reader.readInstance(structure)) != null) {
			updateXXYDist(row);
			xxyDist_.setNoData();
			xxyDist_.xyDist_.setNoData();
		}

		m_BestK_ = K;
		m_BestattIt = nAttributes;

		switch (m_S) {
		case "KDB":
			learnStructureKDB();
			break;
		case "UpperKDB":
			learnStructureUpperKDB();
			break;
		case "LowerKDB":
			this.learnStructureLowerKDB();
			break;
		case "EKDB":
			this.learnStructureAllKDB();
			break;
		case "SKDB":
			learnStructureSKDB(structure, sourceFile);
			break;
		case "UpperSKDB":
			learnStructureUpperSKDB(structure, sourceFile);
			break;
		case "LowerSKDB":
			this.learnStructureLowerSKDB(structure, sourceFile);
			break;
		case "ESKDB":
			this.learnstructureAllSKDB(structure, sourceFile);
			break;
		default:
			System.out.println("value of m_S has to be in set {UpperKDB, SKDB,ESKDB}");
		}
//		parentForEachAtt2 = new ArrayList<HashMap<ArrayList<Integer>,Integer>>();
//		HashMap<ArrayList<Integer>,Integer> temp2 = new HashMap<ArrayList<Integer>,Integer>();
//		
		parentOrderforEachAtt = new ArrayList<ArrayList<ArrayList<Integer>>>();
		ArrayList<ArrayList<Integer>> temp = new ArrayList<ArrayList<Integer>>();
		for (int u = 0; u < this.nAttributes; u++) {
			parentOrderforEachAtt.add(temp);
//			parentForEachAtt2.add(temp2);
		}
		
		// add y as its first parent
		for (int i = 0; i < parentOrder.size(); i++) {
			for (int j = 0; j < parentOrder.get(i).length; j++) {

				// add y as its first parent
				int[][] b = parentOrder.get(i)[j];
				for (int k = 0; k < b.length; k++) {
					int[] c;
					if (b[k] == null) {
						c = new int[1];
						c[0] = structure.classIndex();
					} else {
						c = new int[b[k].length + 1];
						c[0] = structure.classIndex();
						for (int z = 0; z < b[k].length; z++) {
							c[z + 1] = b[k][z];
						}
					}
					parentOrder.get(i)[j][k] = c;
				}
			}
		}
		ArrayList<Integer> a;
		
		// combine all the parents for each attribute, to save memory
		for (int i = 0; i < upperOrder.size(); i++) {
			int[] order = upperOrder.get(i);
			
			for (int j = 0; j < parentOrder.get(i).length; j++) {
				
				temp = (ArrayList<ArrayList<Integer>>) parentOrderforEachAtt.get(order[j]).clone();
				for (int z = 0; z < parentOrder.get(i)[j].length; z++) {
					int[] tempArray = parentOrder.get(i)[j][z];
					if (!isInList(temp, tempArray)){
						ArrayList<Integer> list = new ArrayList<Integer>();
						
						for(int k = 0; k < tempArray.length; k++){
							list.add(tempArray[k]);
						}
						temp.add(list);
					}
						
				}
				parentOrderforEachAtt.set(order[j], temp);
				
//				temp2 = new HashMap<ArrayList<Integer>,Integer>();
//				for (int z = 0; z < parentOrder.get(i)[j].length; z++) {
//					int[] tempArray = parentOrder.get(i)[j][z];
//					a = new ArrayList<Integer>();
//					for(int k = 0; k < tempArray.length; k++){
//						a.add(tempArray[k]);
//					}
//	
//					if(temp2.containsKey(a)){
//						int value = temp2.get(a)+1;
//						temp2.put(a,value);
//					}else{
//						temp2.put(a, 1);
//					}						
//				}
//				
//				parentForEachAtt2.set(order[j], temp2);

			}
		}
//		 printStructure();
	}

	private void learnstructureAllSKDB(Instances structure, File sourceFile) {
		// TODO Auto-generated method stub
		
	}

	private void learnStructureLowerSKDB(Instances structure, File sourceFile) throws FileNotFoundException, IOException {
		// TODO Auto-generated method stub
		int m_KDB = m_BestK_;

		double[] mi = new double[nAttributes];
		double[][] cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// sort attributes on MI with the class
		m_Order = SUtils.sort(mi);

		// Calculate parents based on MI and CMI
		for (int u = 0; u < nAttributes; u++) {
			int nk = Math.min(u, m_KDB);
			if (nk > 0) {
				m_Parents[u] = new int[nk];
				double[] cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}

				int[] cmiOrder = SUtils.sort(cmi_values);

				for (int j = 0; j < nk; j++) {
					m_Parents[u][j] = m_Order[cmiOrder[j]];
				}
			}
		}

		// print the structure
		// System.out.println(Arrays.toString(m_Order));
		// for (int i = 0; i < nAttributes; i++) {
		// System.out.print(i + " : ");
		// if (m_Parents[i] != null) {
		// for (int j = 0; j < m_Parents[i].length; j++) {
		// System.out.print(m_Parents[i][j] + ",");
		// }
		// }
		// System.out.println();
		// }

		// System.out.println("**********************************************");
		// System.out.println("SKDB: First Pass Finished");
		// System.out.println("**********************************************");

		wdBayesParametersTree dParameters_ = new wdBayesParametersTree(nAttributes, nc, paramsPerAtt, m_Order,
				m_Parents, 1);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		Instance instance;
		int N = 0;
		while ((instance = reader.readInstance(structure)) != null) {
			dParameters_.update(instance);
			N++;
		}

		dParameters_.countsToProbability();

		// System.out.println("**********************************************");
		// System.out.println("SKDB: Second Pass Finished");
		// System.out.println("**********************************************");

		double[][] foldLossFunctallK_ = new double[m_KDB + 1][nAttributes + 1];
		double[][] posteriorDist = new double[m_KDB + 1][nc];

		/* Start the third costly pass through the data */
		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		while ((instance = reader.readInstance(structure)) != null) {
			int x_C = (int) instance.classValue();

			for (int y = 0; y < nc; y++) {
				posteriorDist[0][y] = dParameters_.ploocv(y, x_C);
			}
			SUtils.normalize(posteriorDist[0]);

			double error = 1.0 - posteriorDist[0][x_C];
			foldLossFunctallK_[0][nAttributes] += error * error;

			for (int k = 1; k <= m_KDB; k++) {
				for (int y = 0; y < nc; y++) {
					posteriorDist[k][y] = posteriorDist[0][y];
				}
				foldLossFunctallK_[k][nAttributes] += error * error;
			}

			for (int u = 0; u < nAttributes; u++) {
				// Discounting inst from counts
				dParameters_.updateClassDistributionloocv(posteriorDist, u, m_Order[u], instance, m_KDB);

				for (int k = 0; k <= m_KDB; k++)
					SUtils.normalize(posteriorDist[k]);

				for (int k = 0; k <= m_KDB; k++) {
					error = 1.0 - posteriorDist[k][x_C];
					foldLossFunctallK_[k][u] += error * error;
				}
			}
		}

		/* Start the book keeping, select the best k and best attributes */
		// for (int k = 0; k <= m_KDB; k++) {
		// System.out.println("k = " + k);
		// for (int u = 0; u < nAttributes; u++) {
		// System.out.print(foldLossFunctallK_[k][u] + ", ");
		// }
		// System.out.println(foldLossFunctallK_[k][nAttributes]);
		// }

		// Proper kdb selective (RMSE)
		for (int k = 0; k <= m_KDB; k++) {
			for (int att = 0; att < nAttributes + 1; att++) {
				foldLossFunctallK_[k][att] = Math.sqrt(foldLossFunctallK_[k][att] / N);
			}
			// The prior is the same for all values of k_
			foldLossFunctallK_[k][nAttributes] = foldLossFunctallK_[0][nAttributes];
		}

		double globalmin = foldLossFunctallK_[0][nAttributes];

		for (int u = 0; u < nAttributes; u++) {
			for (int k = 0; k <= m_KDB; k++) {
				if (foldLossFunctallK_[k][u] < globalmin) {
					globalmin = foldLossFunctallK_[k][u];
					m_BestattIt = u;
					m_BestK_ = k;
				}
			}
		}

		m_BestattIt += 1;

		if (m_BestattIt > nAttributes)
			m_BestattIt = 0;

		// for (int k = 0; k <= m_KDB; k++) {
		// System.out.println("k = " + k);
		// for (int u = 0; u < nAttributes; u++) {
		// System.out.print(foldLossFunctallK_[k][u] + ", ");
		// }
		// System.out.println(foldLossFunctallK_[k][nAttributes]);
		// }
		// System.out.println("globalmin: "+globalmin);
		 System.out.println("Number of features selected is: " + m_BestattIt +
		 " out of " + nAttributes + " features");
		 System.out.println("best k is: " + m_BestK_);

		// System.out.println("**********************************************");
		// System.out.println("SKDB: Third Pass Finished");
		// System.out.println("**********************************************");

		// Update m_Parents based on m_Order
		int[][] m_ParentsTemp = new int[nAttributes][];
		for (int u = 0; u < nAttributes; u++) {
			if (m_Parents[u] != null) {
				int nK = Math.min(m_Parents[u].length, m_BestK_);
				m_ParentsTemp[u] = new int[nK];

				for (int j = 0; j < nK; j++) {
					m_ParentsTemp[u][j] = m_Parents[u][j];
				}
			}
		}

//		 print the structure
		 System.out.println(Arrays.toString(m_Order));
		 for (int i = 0; i < m_BestattIt; i++) {
		 System.out.print(m_Order[i] + " : ");
		 if (m_Parents[i] != null) {
		 for (int j = 0; j < m_Parents[i].length; j++) {
		 System.out.print(m_Parents[i][j] + ",");
		 }
		 }
		 System.out.println();
		 }

		m_Parents = null;
		m_Parents = m_ParentsTemp;
		m_ParentsTemp = null;

		int[] tempAtt = new int[this.m_BestattIt];
		for (int i = 0; i < m_BestattIt; i++) {
			tempAtt[i] = m_Order[i];
		}
		
		System.out.println(Arrays.toString(tempAtt));
		upperOrder.add(tempAtt);
		
//		for (int i = 0; i < m_BestattIt; i++) {
//			
//			Collections.shuffle(Arrays.asList(m_Parents[i]));
//			System.out.println("shuffle: "+Arrays.toString(m_Parents[i]));
////			tempParent[i][0] = m_Parents[i];
//		}
		
		int[][][] tempParent = new int[m_BestattIt][1][m_BestK_];
		for (int i = 0; i < m_BestattIt; i++) {
			tempParent[i][0] = m_Parents[i];
		}
		this.parentOrder.add(tempParent);

		m_Order = null;
		m_Parents = null;
	}

	private void learnStructureKDB() {

		double[] mi = new double[nAttributes];
		double[][] cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// Sort attributes on MI with the class
		m_Order = SUtils.sort(mi);
//		System.out.println(Arrays.toString(m_Order));
		// Calculate parents based on MI and CMI
		for (int u = 0; u < nAttributes; u++) {

			int nK = Math.min(u, K);

			if (nK > 0) {
				m_Parents[u] = new int[nK];

				double[] cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}
				int[] cmiOrder = SUtils.sort(cmi_values);

				for (int j = 0; j < nK; j++) {
					m_Parents[u][j] = m_Order[cmiOrder[j]];
				}
			}
		}

		// Update m_Parents based on m_Order
		int[][] m_ParentsTemp = new int[nAttributes][];
		for (int u = 0; u < nAttributes; u++) {
			if (m_Parents[u] != null) {
				m_ParentsTemp[m_Order[u]] = new int[m_Parents[u].length];

				for (int j = 0; j < m_Parents[u].length; j++) {
					m_ParentsTemp[m_Order[u]][j] = m_Parents[u][j];
				}
			}
		}

		m_Parents = null;
		m_Parents = m_ParentsTemp;
		m_ParentsTemp = null;

		int[][][] temp = new int[nAttributes][1][];
		for (int i = 0; i < m_Parents.length; i++) {
			temp[i][0] = m_Parents[i];
		}

		parentOrder.add(temp);

		for (int i = 0; i < nAttributes; i++) {
			m_Order[i] = i;
		}
		upperOrder.add(m_Order);

		m_Order = null;
		m_Parents = null;
		temp = null;
	}
	
	private void learnStructureUpperKDB() {

		double[] mi = new double[nAttributes];
		double[][] cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// Sort attributes on MI with the class
		m_Order = SUtils.sort(mi);
		// add the best order to ensemble first
		upperOrder.add(m_Order);
		
		double[] tempMI = new double[mi.length];
		// Sort MI according to MI
		for (int i = 0; i < mi.length; i++) {
			tempMI[i] = mi[m_Order[i]];
		}
		
		// sample for other upper orders
		while(this.upperOrder.size() < ensembleSize){
			double[] miCopy = Arrays.copyOf(tempMI, tempMI.length);
			int[] res = sampleReNormalizing(m_Order, miCopy, miCopy.length);
			upperOrder.add(res);
		}

		// calcute parent order for each upper order
		for (int k = 0; k < upperOrder.size(); k++) {

			int[] tempOrder = upperOrder.get(k).clone();
			// each attribute only has 1 parent order in this algorithm
			int[][][] parentTemp = new int[nAttributes][1][]; 
			for (int u = 0; u < nAttributes; u++) {
				int nK = Math.min(u, K);

				if (nK > 0) {
					parentTemp[u][0] = new int[nK];

					double[] cmi_values = new double[u];
					for (int j = 0; j < u; j++) {
						cmi_values[j] = cmi[tempOrder[u]][tempOrder[j]];
					}

					int[] cmiOrder = SUtils.sort(cmi_values);
					for (int j = 0; j < nK; j++) {
						parentTemp[u][0][j] = tempOrder[cmiOrder[j]];
					}
				}
			}

			// Update parentOrder based on m_Order
			// int[][][] temp1 = new int[nAttributes][1][];
			// for (int u = 0; u < nAttributes; u++) {
			// temp1[tempOrder[u]] = parentTemp[u];
			// }

			parentOrder.add(parentTemp);
			parentTemp = null;
			tempOrder = null;
		}
	}

	private void learnStructureUpperSKDB(Instances structure, File sourceFile)
			throws FileNotFoundException, IOException {
		this.learnStructureUpperKDB();

		int m_KDB = m_BestK_;

		for (int i = 0; i < this.parentOrder.size(); i++) {
			int[] order = upperOrder.get(i);
			int[][] parents = new int[this.nAttributes][0];
			for (int u = 0; u < this.nAttributes; u++) {
				parents[u] = parentOrder.get(i)[u][0];
			}

			// for(int u =0; u < parents.length; u++){
			// System.out.println(Arrays.toString(parents[u]));
			// }

			wdBayesParametersTree dParameters_ = new wdBayesParametersTree(nAttributes, nc, paramsPerAtt, order,
					parents, 1);
			ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
			Instance instance;
			int N = 0;
			while ((instance = reader.readInstance(structure)) != null) {
				dParameters_.update(instance);
				N++;
			}

			dParameters_.countsToProbability();

			double[][] foldLossFunctallK_ = new double[m_KDB + 1][nAttributes + 1];
			double[][] posteriorDist = new double[m_KDB + 1][nc];

			/* Start the third costly pass through the data */
			reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
			while ((instance = reader.readInstance(structure)) != null) {
				int x_C = (int) instance.classValue();

				for (int y = 0; y < nc; y++) {
					posteriorDist[0][y] = dParameters_.ploocv(y, x_C);
				}
				SUtils.normalize(posteriorDist[0]);

				double error = 1.0 - posteriorDist[0][x_C];
				foldLossFunctallK_[0][nAttributes] += error * error;

				for (int k = 1; k <= m_KDB; k++) {
					for (int y = 0; y < nc; y++) {
						posteriorDist[k][y] = posteriorDist[0][y];
					}
					foldLossFunctallK_[k][nAttributes] += error * error;
				}

				for (int u = 0; u < nAttributes; u++) {
					// Discounting inst from counts
					dParameters_.updateClassDistributionloocv(posteriorDist, u, order[u], instance, m_KDB);

					for (int k = 0; k <= m_KDB; k++)
						SUtils.normalize(posteriorDist[k]);

					for (int k = 0; k <= m_KDB; k++) {
						error = 1.0 - posteriorDist[k][x_C];
						foldLossFunctallK_[k][u] += error * error;
					}
				}
			}

			/* Start the book keeping, select the best k and best attributes */
			// for (int k = 0; k <= m_KDB; k++) {
			// System.out.println("k = " + k);
			// for (int u = 0; u < nAttributes; u++) {
			// System.out.print(foldLossFunctallK_[k][u] + ", ");
			// }
			// System.out.println(foldLossFunctallK_[k][nAttributes]);
			// }

			// Proper kdb selective (RMSE)
			for (int k = 0; k <= m_KDB; k++) {
				for (int att = 0; att < nAttributes + 1; att++) {
					foldLossFunctallK_[k][att] = Math.sqrt(foldLossFunctallK_[k][att] / N);
				}
				// The prior is the same for all values of k_
				foldLossFunctallK_[k][nAttributes] = foldLossFunctallK_[0][nAttributes];
			}

			double globalmin = foldLossFunctallK_[0][nAttributes];

			for (int u = 0; u < nAttributes; u++) {
				for (int k = 0; k <= m_KDB; k++) {
					if (foldLossFunctallK_[k][u] < globalmin) {
						globalmin = foldLossFunctallK_[k][u];
						m_BestattIt = u;
						m_BestK_ = k;
					}
				}
			}

			m_BestattIt += 1;
//			m_BestK_ +=1;

			if (m_BestattIt > nAttributes)
				m_BestattIt = 0;

			// for (int k = 0; k <= m_KDB; k++) {
			// System.out.println("k = " + k);
			// for (int u = 0; u < nAttributes; u++) {
			// System.out.print(foldLossFunctallK_[k][u] + ", ");
			// }
			// System.out.println(foldLossFunctallK_[k][nAttributes]);
			// }

//			System.out
//					.println("Number of features selected is: " + m_BestattIt + " out of " + nAttributes + " features");
//			System.out.println("best k is: " + m_BestK_);
			//
			// System.out.println("**********************************************");
			// System.out.println("SKDB: Third Pass Finished");
			// System.out.println("**********************************************");

			// Update m_Parents based on m_Order
			int[][] m_ParentsTemp = new int[nAttributes][];
			for (int u = 0; u < nAttributes; u++) {

				if (parents[u] != null) {
					int nK = Math.min(parents[u].length, m_BestK_);
					m_ParentsTemp[u] = new int[nK];

					for (int j = 0; j < nK; j++) {
						m_ParentsTemp[u][j] = parents[u][j];
					}
				}
			}

			// print the structure
			// System.out.println(Arrays.toString(order));
			// for (int u = 0; u < nAttributes; u++) {
			// System.out.print(u + " : ");
			// if (m_ParentsTemp[u] != null) {
			// for (int j = 0; j < m_ParentsTemp[u].length; j++) {
			// System.out.print(m_ParentsTemp[u][j] + ",");
			// }
			// }
			// System.out.println();
			// }

			parents = null;
			parents = m_ParentsTemp;
			m_ParentsTemp = null;

			int[][][] tempParent = new int[this.m_BestattIt][1][1];
			for (int j = 0; j < m_BestattIt; j++) {
				tempParent[j][0] = parents[j];

			}
			this.parentOrder.set(i, tempParent);

			// for (int u = 0; u < tempParent.length; u++) {
			// System.out.print(order[u] + " : ");
			// if (tempParent[u] != null) {
			// for (int j = 0; j < tempParent[u].length; j++) {
			// System.out.print(Arrays.toString(tempParent[u][j]) + ",");
			// }
			// }
			// System.out.println();
			// }
			//
			//
			int[] tempAtt = new int[this.m_BestattIt];
			for (int j = 0; j < m_BestattIt; j++) {
				tempAtt[j] = order[j];
			}
			// System.out.println(Arrays.toString(order));
			// System.out.println(Arrays.toString(tempAtt));
			upperOrder.set(i, tempAtt);

			tempParent = null;
			tempAtt = null;
			m_Order = null;
			m_Parents = null;
		}
	}

	private void learnStructureAllKDB() {
//		System.out.println("all order");
		double[] mi = new double[nAttributes];
		double[][] cmi = new double[nAttributes][nAttributes];

		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// Sort attributes on MI with the class
		m_Order = SUtils.sort(mi);
		upperOrder.add(m_Order);
		double[] tempMI = new double[mi.length];
		for (int i = 0; i < mi.length; i++) {
			tempMI[i] = mi[m_Order[i]];
		}

		// create multiple upper orders
		while (upperOrder.size() < ensembleSize) {
			double[] miCopy = Arrays.copyOf(tempMI, tempMI.length);
			int[] res = sampleReNormalizing(m_Order, miCopy, miCopy.length);
			upperOrder.add(res);
		}

		// generate parents for each upperOrder
		int[] tempOrder;
		ArrayList<Integer> S;
		int[][][] parentTemp;
		double[] cmi_values;
		int[] cmiOrder;

		for (int k = 0; k < upperOrder.size(); k++) {

			tempOrder = upperOrder.get(k).clone();
			S = new ArrayList<Integer>();
			parentTemp = new int[nAttributes][][];

			for (int u = 0; u < nAttributes; u++) {

				if (u == 0) {
					parentTemp[u] = new int[1][];
				} else if (u > 0 && u <= K) {
					cmi_values = new double[u];
					for (int j = 0; j < u; j++) {
						cmi_values[j] = cmi[tempOrder[u]][tempOrder[j]];
					}

					cmiOrder = SUtils.sort(cmi_values);
					int[] parents = new int[u];
					for (int j = 0; j < u; j++) {
						parents[j] = tempOrder[cmiOrder[j]];
					}

					parentTemp[u] = new int[1][];
					parentTemp[u][0] = parents;
					parents = null;

				} else {// building ensemble
//					int M = K + (int) Math.sqrt(u - K) - 1;
					int M = K  + (int) Math.sqrt(u - K);
					cmi_values = new double[u];
					for (int j = 0; j < u; j++) {
						cmi_values[j] = cmi[tempOrder[u]][tempOrder[j]];
					}
					cmiOrder = SUtils.sort(cmi_values);
					
					//desending sorted CMI and order
					int[] tempOrder111 = new int[S.size()];
					for (int i = 0; i < tempOrder111.length; i++) {
						tempOrder111[i] = S.get(cmiOrder[i]);
					}

					double[] tempCMI = new double[u];
					for (int i = 0; i < u; i++) {
						tempCMI[i] = cmi[tempOrder[u]][tempOrder111[i]];
					}

					// some data set may have more than K attributes, but the
					// CMI
					// could be zero, here process equal to single KDB
					if (Utils.sum(tempCMI) == 0) {

						int[] parents = new int[K];
						for (int j = 0; j < K; j++) {
							parents[j] = tempOrder[cmiOrder[j]];
						}

						parentTemp[u] = new int[1][];
						parentTemp[u][0] = parents;
						parents = null;

					} else {// re-normalizing sampling

						ArrayList<int[]> temp = new ArrayList<int[]>();
//						int size = u < 7 ? Math.min(SUtils.combination(u, M), ensembleSize) : ensembleSize;
						double[] cmi111;
						while (temp.size() < ensembleSize) {

							cmi111 = new double[tempCMI.length];
							System.arraycopy(tempCMI, 0, cmi111, 0, tempCMI.length);

							int[] res = sampleReNormalizing(tempOrder111, cmi111, M);
							// choose the top K out from M
							double[] cmiM = new double[res.length];
							for (int j = 0; j < cmiM.length; j++) {
								cmiM[j] = cmi[tempOrder[u]][res[j]];
							}

							cmiOrder = SUtils.sort(cmiM);

							int[] parents = new int[K];
							for (int i = 0; i < parents.length; i++) {
								parents[i] = res[cmiOrder[i]];
							}

//							if (temp.size() == 0) {
								temp.add(parents);
//							} else {
//								boolean findSame = false;
//								for (int b = 0; b < temp.size(); b++) {
//									int[] a = temp.get(b);
//									if (Arrays.equals(a, parents)) {
//										findSame = true;
//										break;
//									}
//								}
//								if (findSame == false) {
//									temp.add(parents);
//								}
//							}
							parents = null;
						}

						parentTemp[u] = new int[temp.size()][];

						for (int i = 0; i < temp.size(); i++) {
							parentTemp[u][i] = temp.get(i);
						}
					}
				}
				S.add(tempOrder[u]);
			}

			// Update parentOrder based on m_Order
//			int[][][] temp1 = new int[nAttributes][][];
//			for (int u = 0; u < nAttributes; u++) {
//				temp1[tempOrder[u]] = parentTemp[u];
//			}
//
//			parentOrder.add(temp1);
//			temp1 = null;
//			parentTemp = null;
			
			parentOrder.add(parentTemp);

//			for (int i = 0; i < nAttributes; i++) {
//				tempOrder[i] = i;
//			}

			upperOrder.set(k, tempOrder);
			tempOrder = null;
		}
	}

	private void learnStructureLowerKDB() {
		
		double[] mi = new double[nAttributes];
		double[][] cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// Sort attributes on MI with the class
		m_Order = SUtils.sort(mi);
//		System.out.println(Arrays.toString(m_Order));

		int[][][] parentTemp = new int[nAttributes][][];
		double[] cmi_values;
		int[] cmiOrder;

		for (int u = 0; u < this.nAttributes; u++) {
			if (u == 0) {
				parentTemp[u] = new int[1][];
			} else if (u > 0 && u <= K) {

				cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}
				cmiOrder = SUtils.sort(cmi_values);

				int[] parents = new int[u];
				for (int j = 0; j < u; j++) {
					parents[j] = m_Order[cmiOrder[j]];
				}

				parentTemp[u] = new int[1][];
				parentTemp[u][0] = parents;
				parents = null;

			} else {// building ensemble

//				int M = K + (int) Math.sqrt(u - K) - 1;
				int M = K + (int) Math.sqrt(u - K);
				cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}
				cmiOrder = SUtils.sort(cmi_values);

				//desending sorted CMI and order
				int[] tempOrder111 = new int[u];
				for (int i = 0; i < u; i++) {
					tempOrder111[i] = m_Order[cmiOrder[i]];
				}
				double[] tempCMI = new double[u];
				for (int i = 0; i < u; i++) {
					tempCMI[i] = cmi[m_Order[u]][tempOrder111[i]];
				}
				
				// some data set may have more than K attributes, but the CMI
				// could be zero, here process equal to single KDB
				if (Utils.sum(tempCMI) == 0) {

					int[] parents = new int[K];
					for (int j = 0; j < K; j++) {
						parents[j] = m_Order[cmiOrder[j]];
					}

					parentTemp[u] = new int[1][];
					parentTemp[u][0] = parents;

				} else {// re-normalizing sampling

					ArrayList<int[]> temp = new ArrayList<int[]>();
//					int size = Math.min(SUtils.combination(u, M),ensembleSize);

//					int size = u < 7 ? Math.min(SUtils.combination(u, M), ensembleSize) : ensembleSize;

					double[] tempCMI111;
					int[] res;
					double[] cmiM;
					while (temp.size() < ensembleSize) {
						tempCMI111 = new double[tempCMI.length];
						for (int j = 0; j < tempCMI.length; j++) {
							tempCMI111[j] = tempCMI[j];
						}
						
//						System.arraycopy(tempCMI, 0, tempCMI111, 0, tempCMI.length);

						res = sampleReNormalizing(tempOrder111, tempCMI111, M);
						cmiM = new double[res.length];
						for (int j = 0; j < cmiM.length; j++) {
							cmiM[j] = cmi[m_Order[u]][res[j]];
						}

						cmiOrder = SUtils.sort(cmiM);

						int[] parents = new int[K];
						for (int i = 0; i < parents.length; i++) {
							parents[i] = res[cmiOrder[i]];
						}

//						if (temp.size() == 0) {
							temp.add(parents);
//						} else {
//							boolean findSame = false;
//							for (int z = 0; z < temp.size(); z++) {
//								int[] a = temp.get(z);
//								if (Arrays.equals(a, parents)) {
//									findSame = true;
//									break;
//								}
//							}
//							if (findSame == false) {
//								temp.add(parents);
//							}
//						}
						parents = null;
					}
					parentTemp[u] = new int[temp.size()][];

					for (int i = 0; i < temp.size(); i++) {
						parentTemp[u][i] = temp.get(i);
					}
					temp = null;
				}
			}
		}

		// Update m_Parents based on m_Order
		int[][][] temp2 = new int[nAttributes][][];
		for (int u = 0; u < nAttributes; u++) {
			temp2[m_Order[u]] = parentTemp[u];
		}

		parentTemp = temp2;
		temp2 = null;

		for (int i = 0; i < nAttributes; i++) {
			m_Order[i] = i;
		}

		upperOrder.add(m_Order);
		parentOrder.add(parentTemp);

		parentTemp = null;
		m_Order = null;
	}
	private void learnStructureSKDB(Instances structure, File sourceFile) throws FileNotFoundException, IOException {
		int m_KDB = m_BestK_;

		double[] mi = new double[nAttributes];
		double[][] cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

		// sort attributes on MI with the class
		m_Order = SUtils.sort(mi);

		// Calculate parents based on MI and CMI
		for (int u = 0; u < nAttributes; u++) {
			int nk = Math.min(u, m_KDB);
			if (nk > 0) {
				m_Parents[u] = new int[nk];
				double[] cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}

				int[] cmiOrder = SUtils.sort(cmi_values);

				for (int j = 0; j < nk; j++) {
					m_Parents[u][j] = m_Order[cmiOrder[j]];
				}
			}
		}

		// print the structure
		// System.out.println(Arrays.toString(m_Order));
		// for (int i = 0; i < nAttributes; i++) {
		// System.out.print(i + " : ");
		// if (m_Parents[i] != null) {
		// for (int j = 0; j < m_Parents[i].length; j++) {
		// System.out.print(m_Parents[i][j] + ",");
		// }
		// }
		// System.out.println();
		// }

		// System.out.println("**********************************************");
		// System.out.println("SKDB: First Pass Finished");
		// System.out.println("**********************************************");

		wdBayesParametersTree dParameters_ = new wdBayesParametersTree(nAttributes, nc, paramsPerAtt, m_Order,
				m_Parents, 1);

		ArffReader reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		Instance instance;
		int N = 0;
		while ((instance = reader.readInstance(structure)) != null) {
			dParameters_.update(instance);
			N++;
		}

		dParameters_.countsToProbability();

		// System.out.println("**********************************************");
		// System.out.println("SKDB: Second Pass Finished");
		// System.out.println("**********************************************");

		double[][] foldLossFunctallK_ = new double[m_KDB + 1][nAttributes + 1];
		double[][] posteriorDist = new double[m_KDB + 1][nc];

		/* Start the third costly pass through the data */
		reader = new ArffReader(new BufferedReader(new FileReader(sourceFile)), 10000);
		while ((instance = reader.readInstance(structure)) != null) {
			int x_C = (int) instance.classValue();

			for (int y = 0; y < nc; y++) {
				posteriorDist[0][y] = dParameters_.ploocv(y, x_C);
			}
			SUtils.normalize(posteriorDist[0]);

			double error = 1.0 - posteriorDist[0][x_C];
			foldLossFunctallK_[0][nAttributes] += error * error;

			for (int k = 1; k <= m_KDB; k++) {

				for (int y = 0; y < nc; y++) {
					posteriorDist[k][y] = posteriorDist[0][y];
				}
				foldLossFunctallK_[k][nAttributes] += error * error;
			}

			for (int u = 0; u < nAttributes; u++) {
				// Discounting inst from counts
				dParameters_.updateClassDistributionloocv(posteriorDist, u, m_Order[u], instance, m_KDB);

				for (int k = 0; k <= m_KDB; k++)
					SUtils.normalize(posteriorDist[k]);

				for (int k = 0; k <= m_KDB; k++) {
					error = 1.0 - posteriorDist[k][x_C];
					foldLossFunctallK_[k][u] += error * error;
				}
			}
		}

		/* Start the book keeping, select the best k and best attributes */
		// for (int k = 0; k <= m_KDB; k++) {
		// System.out.println("k = " + k);
		// for (int u = 0; u < nAttributes; u++) {
		// System.out.print(foldLossFunctallK_[k][u] + ", ");
		// }
		// System.out.println(foldLossFunctallK_[k][nAttributes]);
		// }

		// Proper kdb selective (RMSE)
		for (int k = 0; k <= m_KDB; k++) {
			for (int att = 0; att < nAttributes + 1; att++) {
				foldLossFunctallK_[k][att] = Math.sqrt(foldLossFunctallK_[k][att] / N);
			}
			// The prior is the same for all values of k_
			foldLossFunctallK_[k][nAttributes] = foldLossFunctallK_[0][nAttributes];
		}

		double globalmin = foldLossFunctallK_[0][nAttributes];

		for (int u = 0; u < nAttributes; u++) {
			for (int k = 0; k <= m_KDB; k++) {
				if (foldLossFunctallK_[k][u] < globalmin) {
					globalmin = foldLossFunctallK_[k][u];
					m_BestattIt = u;
					m_BestK_ = k;
				}
			}
		}

		m_BestattIt += 1;

		if (m_BestattIt > nAttributes)
			m_BestattIt = 0;

		// for (int k = 0; k <= m_KDB; k++) {
		// System.out.println("k = " + k);
		// for (int u = 0; u < nAttributes; u++) {
		// System.out.print(foldLossFunctallK_[k][u] + ", ");
		// }
		// System.out.println(foldLossFunctallK_[k][nAttributes]);
		// }
		// System.out.println("globalmin: "+globalmin);
		// System.out.println("Number of features selected is: " + m_BestattIt +
		// " out of " + nAttributes + " features");
		// System.out.println("best k is: " + m_BestK_);

		// System.out.println("**********************************************");
		// System.out.println("SKDB: Third Pass Finished");
		// System.out.println("**********************************************");

		// Update m_Parents based on m_Order
		int[][] m_ParentsTemp = new int[nAttributes][];
		for (int u = 0; u < nAttributes; u++) {
			if (m_Parents[u] != null) {
				int nK = Math.min(m_Parents[u].length, m_BestK_);
				m_ParentsTemp[u] = new int[nK];

				for (int j = 0; j < nK; j++) {
					m_ParentsTemp[u][j] = m_Parents[u][j];
				}
			}
		}

//		 print the structure
		 System.out.println(Arrays.toString(m_Order));
		 for (int i = 0; i < nAttributes; i++) {
		 System.out.print(i + " : ");
		 if (m_Parents[i] != null) {
		 for (int j = 0; j < m_Parents[i].length; j++) {
		 System.out.print(m_Parents[i][j] + ",");
		 }
		 }
		 System.out.println();
		 }

		m_Parents = null;
		m_Parents = m_ParentsTemp;
		m_ParentsTemp = null;

		int[] tempAtt = new int[this.m_BestattIt];
		for (int i = 0; i < m_BestattIt; i++) {
			tempAtt[i] = m_Order[i];
		}
		upperOrder.add(tempAtt);

		int[][][] tempParent = new int[m_BestattIt][1][m_BestK_];
		for (int i = 0; i < m_BestattIt; i++) {
			tempParent[i][0] = m_Parents[i];
		}
		this.parentOrder.add(tempParent);

		m_Order = null;
		m_Parents = null;
	}

	private void updateXXYDist(Instance instance) {
		xxyDist_.update(instance);
	}

	private int cumulativeProbability(double[] array, double p) {

		double cumulativeProbability = 0.0;
		int index = 0;
		for (; index < array.length; index++) {
			cumulativeProbability += array[index];
			if (p <= cumulativeProbability && array[index] != 0) {
				return index;
			}
		}
		return index;
	}

	private int[] sampleReNormalizing(int[] tempS, double[] tempCMI, int M) {
		int[] res = new int[M];
		// System.out.println("\n"+M+" "+ tempCMI.length);
		for (int i = 0; i < M; i++) {

			Utils.normalize(tempCMI);
			// System.out.println(Arrays.toString(tempCMI));
			double p = Math.random();
			// System.out.println("p: "+p);
			int index = cumulativeProbability(tempCMI, p);
			// System.out.println("index"+index+"\t"+tempS[index]);
			res[i] = tempS[index];

			tempCMI[index] = 0;// set the selected probability to be zero, then
								// select another parent again
			// System.out.println("tempCMI"+Arrays.toString(tempCMI));

			if (Utils.sum(tempCMI) == 0) {
				break;
			}
		}
		return res;
	}

	public xxyDist get_XXYDist() {
		return xxyDist_;
	}

	public void setMaxNFreeParams(long maxNFreeParams) {
		this.maxNFreeParams = maxNFreeParams;
	}

	public int[] get_Order() {
		return m_Order;
	}

	public int[][] get_Parents() {
		return m_Parents;
	}

	public ArrayList<int[]> getUpperOrder() {
		return this.upperOrder;
	}

	public ArrayList<int[][][]> getLowerOrder() {
		return parentOrder;
	}

	public ArrayList<ArrayList<ArrayList<Integer>>> getParentOrderforEachAtt() {
		return parentOrderforEachAtt;
	}

	public int get_BestattIt() {
		return m_BestattIt;
	}

	private void printStructure() {

		for (int i = 0; i < parentOrder.size(); i++) {
			System.out.println("for upperorder" + i + ":" + Arrays.toString(upperOrder.get(i)));
			for (int j = 0; j < parentOrder.get(i).length; j++) {
				System.out.print(upperOrder.get(i)[j] + "\t");
				for (int z = 0; z < parentOrder.get(i)[j].length; z++) {
					System.out.print(Arrays.toString(parentOrder.get(i)[j][z]) + "\t");
				}
				System.out.println();
			}
			System.out.println();
		}
		
			
//		System.out.println("print parent for each attribute here");
//		for (int i = 0; i < parentForEachAtt2.size(); i++) {
//			System.out.println("parent for attribute "+i);
//			for(Map.Entry m:parentForEachAtt2.get(i).entrySet()){
//				System.out.print(m.getKey()+"\t"+m.getValue() + "\t");
//			}
//			System.out.println();
//		}
	}

	public static boolean isInList(ArrayList<ArrayList<Integer>> list, int[] candidate) {
		
		for (final ArrayList<Integer> item : list) {
			// convert arraylist to int[]
			int[] temp = new int[item.size()];
			for(int i = 0; i < temp.length; i++){
				temp[i] = item.get(i).intValue();
			}
			
			if (Arrays.equals(temp, candidate)) {
				return true;
			}
		}
		return false;
	}

	public void setEnsembleSize(int size) {
		this.ensembleSize = size;
	}
}
