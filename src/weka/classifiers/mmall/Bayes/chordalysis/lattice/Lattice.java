/*******************************************************************************
 * Copyright (C) 2014 Francois Petitjean
 * 
 * This file is part of Chordalysis.
 * 
 * Chordalysis is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * Chordalysis is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Chordalysis.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package weka.classifiers.mmall.Bayes.chordalysis.lattice;

import java.io.IOException;
import java.util.BitSet;
import java.util.TreeSet;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

/**
 * Represents the lattice over the variables. This lattice is used to access the
 * high-dimensional matrices for different combinations of the variables of a
 * dataset.
 */
public class Lattice {

    LatticeNode all;
    LatticeNode[] singleNodes;
    int nbVariables;
    int nbInstances;

    public Lattice(Instances dataset, boolean missingAsAdditionalValue) {
	init(dataset, missingAsAdditionalValue, false);
    }

    public Lattice(Instances dataset, boolean missingAsAdditionalValue, boolean ignoreLastAttribute) {
	init(dataset, missingAsAdditionalValue, ignoreLastAttribute);
    }

    protected void init(Instances dataset, boolean missingAsAdditionalValue, boolean ignoreLastAttribute) {
	// ~ initialise internal structure for counting (TID sets)
	this.nbInstances = dataset.numInstances();
	this.nbVariables = dataset.numAttributes();
	if (ignoreLastAttribute) {
	    this.nbVariables--;
	}

	BitSet[][] presence = new BitSet[nbVariables][];

	TreeSet<Integer> allAttributesNumbers = new TreeSet<Integer>();
	int[] nbValuesForAttribute = new int[nbVariables];
	for (int a = 0; a < nbVariables; a++) {
	    if (missingAsAdditionalValue) {
		nbValuesForAttribute[a] = dataset.attribute(a).numValues() + 1; // +1
		// for
		// missing
	    } else {
		nbValuesForAttribute[a] = dataset.attribute(a).numValues();
	    }
	    presence[a] = new BitSet[nbValuesForAttribute[a]];
	    allAttributesNumbers.add(a);
	    for (int v = 0; v < presence[a].length; v++) {
		presence[a][v] = new BitSet();
	    }
	}

	for (int i = 0; i < nbInstances; i++) {
	    Instance row = dataset.instance(i);
	    for (int a = 0; a < nbVariables; a++) {
		int indexOfValue;
		if (row.isMissing(a)) {
		    if (missingAsAdditionalValue) {
			// missing at the end
			indexOfValue = dataset.attribute(a).numValues();
		    } else {
			indexOfValue = (int) dataset.meanOrMode(a);
		    }
		} else {
		    String value = row.stringValue(a);
		    indexOfValue = row.attribute(a).indexOfValue(value);
		}
		presence[a][indexOfValue].set(i);

	    }
	}

	// initialise the first nodes of the lattice (i.e., the ones
	// corresponding to single variables
	this.all = new LatticeNode(this, nbValuesForAttribute);
	this.singleNodes = new LatticeNode[nbVariables];
	for (int a = 0; a < nbVariables; a++) {
	    int[] variablesNumbers = { a };
	    LatticeNode node = new LatticeNode(this, variablesNumbers, nbValuesForAttribute, presence[a], all);
	    singleNodes[a] = node;
	}
    }

    protected void init(Instances structure, ArffReader loader, boolean missingAsAdditionalValue, double samplingRate, boolean ignoreLastAttribute,
	    RandomGenerator rg) throws IOException {
	if (rg == null) {
	    rg = new MersenneTwister(3071980);
	}
	// ~ initialise internal structure for counting (TID sets)
	this.nbInstances = 0;
	this.nbVariables = structure.numAttributes();
	if (ignoreLastAttribute) {
	    this.nbVariables--;
	}
	BitSet[][] presence = new BitSet[nbVariables][];

	TreeSet<Integer> allAttributesNumbers = new TreeSet<Integer>();
	int[] nbValuesForAttribute = new int[nbVariables];
	for (int a = 0; a < nbVariables; a++) {
	    if (missingAsAdditionalValue) {
		nbValuesForAttribute[a] = structure.attribute(a).numValues() + 1;// +1
		// for
		// missing
	    } else {
		nbValuesForAttribute[a] = structure.attribute(a).numValues();
	    }
	    presence[a] = new BitSet[nbValuesForAttribute[a]];
	    allAttributesNumbers.add(a);
	    for (int v = 0; v < presence[a].length; v++) {
		presence[a][v] = new BitSet();
	    }
	}

	Instance row;
	while ((row = loader.readInstance(structure)) != null) {
	    if (rg.nextDouble() < samplingRate) {
		boolean skipRow = false;
		for (int a = 0; a < nbVariables; a++) {
		    int indexOfValue;
		    if (row.isMissing(a)) {
			if (missingAsAdditionalValue) {
			    indexOfValue = structure.attribute(a).numValues();// missing
			    // at
			    // the
			    // end
			} else {
			    System.err.println("Don't know what to do with missing without having the entire dataset; ignoring whole row");
			    skipRow = true;
			    break;
			}
		    } else {
			String value = row.stringValue(a);
			indexOfValue = row.attribute(a).indexOfValue(value);
		    }
		    presence[a][indexOfValue].set(this.nbInstances);

		}
		if (!skipRow) {
		    this.nbInstances++;
		}
	    }
	}

	// initialise the first nodes of the lattice (i.e., the ones
	// corresponding to single variables
	this.all = new LatticeNode(this, nbValuesForAttribute);
	this.singleNodes = new LatticeNode[nbVariables];
	for (int a = 0; a < nbVariables; a++) {
	    int[] variablesNumbers = { a };
	    LatticeNode node = new LatticeNode(this, variablesNumbers, nbValuesForAttribute, presence[a], all);
	    singleNodes[a] = node;
	}
    }

    public Lattice(Instances dataset) {
	init(dataset, true, false);
    }

    public Lattice(BitSet[][] presence, int nbInstances) {

	// ~ initialise internal structure for counting (TID sets)
	this.nbInstances = nbInstances;
	this.nbVariables = presence.length;

	int[] nbValuesForAttribute = new int[nbVariables];
	for (int a = 0; a < nbVariables; a++) {
	    nbValuesForAttribute[a] = presence[a].length;
	}

	// initialise the first nodes of the lattice (i.e., the ones
	// corresponding to single variables
	this.all = new LatticeNode(this, nbValuesForAttribute);
	this.singleNodes = new LatticeNode[nbVariables];
	for (int a = 0; a < nbVariables; a++) {
	    int[] variablesNumbers = { a };
	    LatticeNode node = new LatticeNode(this, variablesNumbers, nbValuesForAttribute, presence[a], all);
	    singleNodes[a] = node;
	}

    }

    public Lattice(Instances structure, ArffReader loader) throws IOException {
	init(structure, loader, true, 1.0, false,null);
    }

    public Lattice(Instances structure, ArffReader loader, boolean treatMissingAsAdditionalValue) throws IOException {
	init(structure, loader, treatMissingAsAdditionalValue, 1.0, false,null);
    }

    public Lattice(Instances structure, ArffReader loader, boolean treatMissingAsAdditionalValue, double samplingRate) throws IOException {
	init(structure, loader, treatMissingAsAdditionalValue, samplingRate, false,null);
    }

    public Lattice(Instances structure, ArffReader loader, boolean treatMissingAsAdditionalValue, double samplingRate, boolean ignoreLastAttribute,
	    RandomGenerator rg) throws IOException {
	init(structure, loader, treatMissingAsAdditionalValue, samplingRate, ignoreLastAttribute, rg);
    }

    /**
     * Get a node of the lattice from its integer set representation (e.g.,
     * {0,4,8} for the node representing the correlation of variables 0,4 and 8
     * in the dataset).
     * 
     * @param clique
     *            the list of variable
     * @return the node of the lattice
     */
    public LatticeNode getNode(BitSet clique) {
	int[] variables = new int[clique.cardinality()];
	int current = 0;
	for (int i = clique.nextSetBit(0); i >= 0; i = clique.nextSetBit(i + 1)) {
	    variables[current] = i;
	    current++;
	}
	return getNode(variables);
    }

    /**
     * Get a node of the lattice from its integer set representation -- sorted
     * array of integers (e.g., [0,4,8] for the node representing the
     * correlation of variables 0,4 and 8 in the dataset).
     * 
     * @param variables
     *            the list of variable
     * @return the node of the lattice
     */
    public LatticeNode getNode(int[] variables) {
	LatticeNode node = singleNodes[variables[0]];

	for (int i = 1; i < variables.length; i++) {
	    node = node.getChild(variables[i], this);

	}

	return node;
    }

    /**
     * 
     * @return the number of variables that are modelled by this lattice.
     */
    public int getNbVariables() {
	return this.nbVariables;
    }

    public int getNbInstances() {
	return this.nbInstances;
    }

    protected BitSet getSetForVariable(int variableIndex, int valueIndex) {
	return singleNodes[variableIndex].getSet(valueIndex);
    }

    protected BitSet getSetForPairOfVariables(int variableIndex1, int valueIndex1, int variableIndex2, int valueIndex2) {
	LatticeNode pairNode = singleNodes[variableIndex1].getChild(variableIndex2, this);
	return pairNode.getSet(valueIndex1, valueIndex2);
    }
}
