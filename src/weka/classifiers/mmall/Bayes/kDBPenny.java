/*
 * MMaLL: An open source system for learning from very large data
 * Copyright (C) 2014 Nayyar A Zaidi, Francois Petitjean and Geoffrey I Webb
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Nayyar Zaidi <nayyar.zaidi@monash.edu>
 */

/*
 * wdBayes Classifier
 * 
 * wdBayes.java     
 * Code written by: Nayyar Zaidi, Francois Petitjean
 * 
 * Options:
 * -------
 * 
 * -t /Users/nayyar/WData/datasets_DM/shuttle.arff
 * -V -M 
 * -S "chordalysis"
 * -K 2
 * -P "wCCBN"
 * -W 1
 * 
 */
package weka.classifiers.mmall.Bayes;

import java.util.Arrays;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesNode;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesParametersTree;
import weka.classifiers.mmall.Utils.CorrelationMeasures;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.supervised.attribute.Discretize;

public class kDBPenny extends AbstractClassifier implements OptionHandler {

	/**
     * 
     */
    private static final long serialVersionUID = 691858787988950382L;

	int nInstances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt;
	xxyDist xxyDist_;

	private Discretize m_Disc = null;

	public wdBayesParametersTree dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private int m_KDB = 1; // -K

	private boolean m_MVerb = false; // -V
	private boolean m_Discretization = false; // -D 

	@Override
	public void buildClassifier(Instances m_Instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(m_Instances);

		// Discretize instances if required
		if (m_Discretization) {
			m_Disc = new Discretize();
			m_Disc.setInputFormat(m_Instances);
			m_Instances = weka.filters.Filter.useFilter(m_Instances, m_Disc);
		}

		// remove instances with missing class
		m_Instances.deleteWithMissingClass();
		nAttributes = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();

		nInstances = m_Instances.numInstances();

		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
		}

		m_Parents = new int[nAttributes][];
		m_Order = new int[nAttributes];
		for (int i = 0; i < nAttributes; i++) {
			getM_Order()[i] = i;
		}

		double[] mi = null;
		double[][] cmi = null;

		/*
		 * Initialize kdb structure
		 */

			xxyDist_ = new xxyDist(m_Instances);
			xxyDist_.addToCount(m_Instances);

			mi = new double[nAttributes];
			cmi = new double[nAttributes][nAttributes];
			CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);
			CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);

			// Sort attributes on MI with the class
			m_Order = SUtils.sort(mi);

			// Calculate parents based on MI and CMI
			for (int u = 0; u < nAttributes; u++) {
				int nK = Math.min(u, m_KDB);
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
			for (int i = 0; i < nAttributes; i++) {
				getM_Order()[i] = i;
			}
			m_Parents = null;
			m_Parents = m_ParentsTemp;
			m_ParentsTemp = null;

			// Print the structure
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


			// MAP
			dParameters_ = new wdBayesParametersTree(nAttributes, nc, paramsPerAtt, m_Order, m_Parents, 1);

			// Update dTree_ based on parents
			for (int i = 0; i < nInstances; i++) {
				Instance instance = m_Instances.instance(i);
				dParameters_.update(instance);
			}

			// Convert counts to Probability
			xxyDist_.countsToProbs();
			dParameters_.countsToProbability();

			//System.out.print(dParameters_.getNLL_MAP(m_Instances, getXxyDist_().xyDist_) + ", ");
			System.out.print(dParameters_.getNLL_MAP(m_Instances) + ", ");
			
			dParameters_.printProbabilities();

//			System.out.println();
//			for (int c = 0; c < nc; c++) {
//				System.out.print(xxyDist_.xyDist_.getClassCount(c) + ", ");
//			}
//			for (int u = 0; u < nAttributes; u++) {
//				for (int uval = 0; uval < paramsPerAtt[u]; uval++) {
//					for (int c = 0; c < nc; c++) {
//						System.out.print("P(x_" + u + "=" + uval + " | y = " + c + ") = " + xxyDist_.xyDist_.getCount(u, uval, c) + ",   " );
//					}
//					System.out.println();
//				}
//			}


		// free up some space
		m_Instances = null;
	}

	@Override
	public double[] distributionForInstance(Instance instance) {

		double[] probs = null;

			// MAP
			probs = logDistributionForInstance_MAP(instance);

		SUtils.exp(probs);
		return probs;
	}

	public double[] logDistributionForInstance_MAP(Instance instance) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = xxyDist_.xyDist_.pp(c);
		}

		for (int u = 0; u < nAttributes; u++) {
			wdBayesNode bNode = dParameters_.getBayesNode(instance, u);
			for (int c = 0; c < nc; c++) {
				probs[c] += bNode.getXYCount((int) instance.value(m_Order[u]),	c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}


	// ----------------------------------------------------------------------------------
	// Weka Functions
	// ----------------------------------------------------------------------------------

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		// instances
		result.setMinimumNumberInstances(0);
		return result;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		m_MVerb = Utils.getFlag('V', options);
		m_Discretization = Utils.getFlag('D', options);

		String MK = Utils.getOption('K', options);
		if (MK.length() != 0) {
			m_KDB = Integer.parseInt(MK);
		}

		Utils.checkForRemainingOptions(options);
	}

	@Override
	public String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	public static void main(String[] argv) {
		runClassifier(new kDBPenny(), argv);
	}

	public int getNInstances() {
		return nInstances;
	}

	public int getNc() {
		return nc;
	}

	public xxyDist getXxyDist_() {
		return xxyDist_;
	}

	public wdBayesParametersTree getdParameters_() {
		return dParameters_;
	}

	public int[] getM_Order() {
		return m_Order;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}

	public int getnAttributes() {
		return nAttributes;
	}

}
