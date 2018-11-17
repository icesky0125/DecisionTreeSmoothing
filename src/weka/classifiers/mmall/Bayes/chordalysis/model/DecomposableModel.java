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
package weka.classifiers.mmall.Bayes.chordalysis.model;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.commons.math3.util.FastMath;
import org.jgrapht.experimental.dag.DirectedAcyclicGraph;
import org.jgrapht.experimental.dag.DirectedAcyclicGraph.CycleFoundException;
import org.jgrapht.graph.DefaultEdge;

import weka.classifiers.mmall.Bayes.chordalysis.graph.ChordalGraph;
import weka.classifiers.mmall.Bayes.chordalysis.graph.CliqueGraphEdge;
import weka.classifiers.mmall.Bayes.chordalysis.model.GraphAction.ActionType;
import weka.classifiers.mmall.Bayes.chordalysis.stats.EntropyComputer;
import weka.classifiers.mmall.Bayes.chordalysis.stats.MyPriorityQueue;
import weka.classifiers.mmall.Bayes.chordalysis.stats.NFreeParamsEstimator;
import weka.classifiers.mmall.Bayes.chordalysis.stats.SquaredL2NormComputer;

/**
 * This class makes it possible to represent a decomposable log-linear model. A
 * decomposable log-linear model is linked to a chordal graph representation.
 */
public class DecomposableModel {
    /**
     * The graph representing the decomposable log-linear model
     */
    public ChordalGraph graph;
    TreeMap<GraphAction, List<GraphAction>> actionsForInteraction;
    Double entropy;
    Double encodingLength;
    boolean entropyComputed = false;
    long nbParameters = -1;
    int[] dimensionsForVariables;

    /**
     * Creates a new decomposable log-linear model, considering no interactions
     * (but independence).
     * 
     * @param variables
     *            list of the variables named after their number
     * @param dimensionsForVariables
     *            the number of possible values for every variable
     */
    public DecomposableModel(int[] variables, int[] dimensionsForVariables) {
	graph = new ChordalGraph();
	for (int var : variables) {
	    graph.addVertex(var);
	}
	graph.initStructures();
	this.actionsForInteraction = new TreeMap<GraphAction, List<GraphAction>>();
	for (int i = 0; i < variables.length; i++) {
	    for (int j = i + 1; j < variables.length; j++) {
		Couple<Integer> edge = new Couple<Integer>(i, j);
		ArrayList<GraphAction> actionsList = new ArrayList<GraphAction>();
		GraphAction action = new GraphAction(ActionType.ADD, edge);
		actionsList.add(action);
		this.actionsForInteraction.put(action, actionsList);
	    }
	}

	this.dimensionsForVariables = dimensionsForVariables;
    }

    /**
     * Creates a new decomposable log-linear model by copy from another model.
     * 
     * @param model
     *            the model to copy
     */
    public DecomposableModel(DecomposableModel model) {
	this.graph = (ChordalGraph) model.graph.clone();
	this.actionsForInteraction = null;
	this.entropy = model.entropy;
	this.entropyComputed = model.entropyComputed;
	this.dimensionsForVariables = model.dimensionsForVariables;

    }

    /**
     * Performs a modification of the graph (add an edge to the graph).
     * 
     * @param actionToPerform
     *            the action to perform
     * @param the
     *            model from which the current one has been created
     */
    public void performAction(GraphAction actionToPerform, DecomposableModel fromModel) {
	List<GraphAction> actionsToPerform = fromModel.actionsForInteraction.get(actionToPerform);
	for (GraphAction action : actionsToPerform) {
	    switch (action.type) {
	    case ADD:
		graph.addSecuredEdge(action.getV1(), action.getV2());
		break;
	    case REMOVE:
		graph.removeSecuredEdge(action.getV1(), action.getV2());
		break;
	    default:
		break;
	    }
	}
	entropyComputed = false;
	entropy = Double.NaN; // invalidate the entropy
	nbParameters = -1; // invalidate the nb degrees of freedom
    }

    public void performAction(GraphAction actionToPerform, DecomposableModel fromModel, MyPriorityQueue pq) {
	List<GraphAction> actionsToPerform = fromModel.actionsForInteraction.get(actionToPerform);
	for (GraphAction action : actionsToPerform) {
	    switch (action.type) {
	    case ADD:
		graph.addSecuredEdge(action.getV1(), action.getV2(), pq);
		break;
	    case REMOVE:
		// graph.removeSecuredEdge(action.getV1(),
		// action.getV2(),pq,scorer);
		break;
	    default:
		break;
	    }
	}
	entropyComputed = false;
	entropy = Double.NaN; // invalidate the entropy
	nbParameters = -1; // invalidate the nb degrees of freedom

    }

    /**
     * Computes the available modifications of the graph to keep it triangulated
     */
    public void computeAvailableModifications() {
	computeAvailableModifications(false);
    }

    /**
     * Computes the available modifications of the graph to keep it triangulated
     * 
     * @param backward
     *            set to true if want to include the backward modifications
     */
    public void computeAvailableModifications(boolean backward) {
	actionsForInteraction = new TreeMap<GraphAction, List<GraphAction>>();

	for (int v1 = 0; v1 < dimensionsForVariables.length; v1++) {
	    for (int v2 = v1 + 1; v2 < dimensionsForVariables.length; v2++) {
		// can't add an already present edge
		ArrayList<GraphAction> actionsList = new ArrayList<GraphAction>();
		GraphAction currentAction = null;
		if (graph.isEdgeAddable(v1, v2)) {
		    currentAction = new GraphAction(ActionType.ADD, v1, v2);
		    actionsList.add(currentAction);
		}
		if (!actionsList.isEmpty()) {
		    actionsForInteraction.put(currentAction, actionsList);
		}

		// if (graph.isEdgeAddable(v1, v2)) { // if not already in
		// currentAction = new GraphAction(ActionType.ADD, v1, v2);
		// actionsList.add(currentAction);
		// actionsForInteraction.put(currentAction, actionsList);
		// }
	    }
	}

    }

    /**
     * @return the list of modifications of the graph that can be performed
     */
    public TreeSet<GraphAction> getAvailableInteractions() {
	return new TreeSet<GraphAction>(actionsForInteraction.keySet());
    }

    public Double entropyDiffIfAdding(Integer a, Integer b, EntropyComputer computer) {
	return entropyDiffIfAdding(a, b, computer, false);
    }

    /**
     * Compute the difference in the entropy from this model, to one that would
     * add vertex1 and vertex2 to it
     * 
     * @param a
     * @param b
     * @param computer
     * @return
     */
    public Double entropyDiffIfAdding(Integer a, Integer b, EntropyComputer computer, boolean verbose) {
	// System.out.println("computing actual entropy");
	BitSet Sab = graph.getSeparator(a, b);
	BitSet Sabua = (BitSet) Sab.clone();
	BitSet Sabub = (BitSet) Sab.clone();
	BitSet Sabuaub = (BitSet) Sab.clone();
	Sabua.set(a);
	Sabub.set(b);
	Sabuaub.set(a);
	Sabuaub.set(b);

	Double entropy = 0.0;
	Double tmp;

	// Sab
	tmp = computer.computeEntropy(Sab);
	if (tmp == null) {
	    entropy = null;
	    return entropy;
	} else {
	    entropy -= tmp;
	}
	if (verbose)
	    System.out.println("-" + Sab + ":" + tmp);

	// Sab + a
	tmp = computer.computeEntropy(Sabua);
	if (tmp == null) {
	    entropy = null;
	    return entropy;
	} else {
	    entropy += tmp;
	}
	if (verbose)
	    System.out.println("+" + Sabua + ":" + tmp);

	// Sab + b
	tmp = computer.computeEntropy(Sabub);
	if (tmp == null) {
	    entropy = null;
	    return entropy;
	} else {
	    entropy += tmp;
	}
	if (verbose)
	    System.out.println("+" + Sabub + ":" + tmp);

	// Sab + a + b
	tmp = computer.computeEntropy(Sabuaub);
	if (tmp == null) {
	    entropy = null;
	    return entropy;
	} else {
	    entropy -= tmp;
	}
	if (verbose)
	    System.out.println("-" + Sabuaub + ":" + tmp);

	return entropy;
    }
    
    /**
     * Compute the difference in the entropy from this model, to one that would
     * add vertex1 and vertex2 to it
     * 
     * @param a
     * @param b
     * @param computer
     * @return
     */
    public Double regularizationDiffIfAdding(Integer a, Integer b, SquaredL2NormComputer computer, boolean verbose) {
	// System.out.println("computing actual entropy");
	BitSet Sab = graph.getSeparator(a, b);
	BitSet Sabua = (BitSet) Sab.clone();
	BitSet Sabub = (BitSet) Sab.clone();
	BitSet Sabuaub = (BitSet) Sab.clone();
	Sabua.set(a);
	Sabub.set(b);
	Sabuaub.set(a);
	Sabuaub.set(b);

	Double reg = 0.0;
	Double tmp;

	// Sab
	tmp = computer.computeNorm(Sab);
	if (tmp == null) {
	    reg = null;
	    return reg;
	} else {
	    reg += tmp;
	}
	if (verbose)
	    System.out.println("+" + Sab + ":" + tmp);

	// Sab + a
	tmp = computer.computeNorm(Sabua);
	if (tmp == null) {
	    reg = null;
	    return reg;
	} else {
	    reg -= tmp;
	}
	if (verbose)
	    System.out.println("-" + Sabua + ":" + tmp);

	// Sab + b
	tmp = computer.computeNorm(Sabub);
	if (tmp == null) {
	    reg = null;
	    return reg;
	} else {
	    reg -= tmp;
	}
	if (verbose)
	    System.out.println("-" + Sabub + ":" + tmp);

	// Sab + a + b
	tmp = computer.computeNorm(Sabuaub);
	if (tmp == null) {
	    reg = null;
	    return reg;
	} else {
	    reg += tmp;
	}
	if (verbose)
	    System.out.println("+" + Sabuaub + ":" + tmp);

	return reg;
    }

    public int treeWidthIfAdding(Integer a, Integer b) {
	// System.out.println("computing actual entropy");
	BitSet Sab = graph.getSeparator(a, b);
	BitSet Sabuaub = (BitSet) Sab.clone();
	Sabuaub.set(a);
	Sabuaub.set(b);
	return Sabuaub.cardinality();

    }

    public int treeWidth() {
	return graph.getTreeWidth();
    }

    /**
     * Computes the number of parameters in the new clique (Cab) that would be
     * formed if (a,b) was added to the graph.
     * 
     * @param a
     * @param b
     * @return
     */
    public int getNumberParametersCabIfAdding(Integer a, Integer b) {
	// System.out.println("computing actual entropy");
	BitSet Sab = graph.getSeparator(a, b);
	BitSet Sabuaub = (BitSet) Sab.clone();
	Sabuaub.set(a);
	Sabuaub.set(b);

	// Sab + a + b
	int tmpNBDF = 1;
	for (int var = Sabuaub.nextSetBit(0); var >= 0; var = Sabuaub.nextSetBit(var + 1)) {
	    tmpNBDF *= dimensionsForVariables[var];
	}
	tmpNBDF = tmpNBDF - 1;
	return tmpNBDF;

    }

    public long nbParametersDiffIfAdding(Integer a, Integer b) {
	// System.out.println("computing actual entropy");
	BitSet Sab = graph.getSeparator(a, b);
	BitSet Sabua = (BitSet) Sab.clone();
	BitSet Sabub = (BitSet) Sab.clone();
	BitSet Sabuaub = (BitSet) Sab.clone();
	Sabua.set(a);
	Sabub.set(b);
	Sabuaub.set(a);
	Sabuaub.set(b);

	long diffNbParameters = 0;

	// Sab
	int tmpNBDF = 1;
	for (int var = Sab.nextSetBit(0); var >= 0; var = Sab.nextSetBit(var + 1)) {
	    tmpNBDF *= dimensionsForVariables[var];
	}
	tmpNBDF = tmpNBDF - 1;
	diffNbParameters += tmpNBDF;

	// Sab + a
	tmpNBDF = 1;
	for (int var = Sabua.nextSetBit(0); var >= 0; var = Sabua.nextSetBit(var + 1)) {
	    tmpNBDF *= dimensionsForVariables[var];
	}
	tmpNBDF = tmpNBDF - 1;
	diffNbParameters -= tmpNBDF;

	// Sab + a + b
	tmpNBDF = 1;
	for (int var = Sabuaub.nextSetBit(0); var >= 0; var = Sabuaub.nextSetBit(var + 1)) {
	    tmpNBDF *= dimensionsForVariables[var];
	}
	tmpNBDF = tmpNBDF - 1;
	diffNbParameters += tmpNBDF;

	// Sab + b
	tmpNBDF = 1;
	for (int var = Sabub.nextSetBit(0); var >= 0; var = Sabub.nextSetBit(var + 1)) {
	    tmpNBDF *= dimensionsForVariables[var];
	}
	tmpNBDF = tmpNBDF - 1;
	diffNbParameters -= tmpNBDF;

	return diffNbParameters;
    }

    public long nbParametersDiffIfAdding(Integer a, Integer b, BitSet Sab) {
	// System.out.println("computing actual entropy");
	BitSet Sabua = (BitSet) Sab.clone();
	BitSet Sabub = (BitSet) Sab.clone();
	BitSet Sabuaub = (BitSet) Sab.clone();
	Sabua.set(a);
	Sabub.set(b);
	Sabuaub.set(a);
	Sabuaub.set(b);

	long diffNbParameters = 0;

	// Sab
	int tmpNBDF = 1;
	for (int var = Sab.nextSetBit(0); var >= 0; var = Sab.nextSetBit(var + 1)) {
	    tmpNBDF *= dimensionsForVariables[var];
	}
	tmpNBDF = tmpNBDF - 1;
	diffNbParameters += tmpNBDF;

	// Sab + a
	tmpNBDF = 1;
	for (int var = Sabua.nextSetBit(0); var >= 0; var = Sabua.nextSetBit(var + 1)) {
	    tmpNBDF *= dimensionsForVariables[var];
	}
	tmpNBDF = tmpNBDF - 1;
	diffNbParameters -= tmpNBDF;

	// Sab + a + b
	tmpNBDF = 1;
	for (int var = Sabuaub.nextSetBit(0); var >= 0; var = Sabuaub.nextSetBit(var + 1)) {
	    tmpNBDF *= dimensionsForVariables[var];
	}
	tmpNBDF = tmpNBDF - 1;
	diffNbParameters += tmpNBDF;

	// Sab + b
	tmpNBDF = 1;
	for (int var = Sabub.nextSetBit(0); var >= 0; var = Sabub.nextSetBit(var + 1)) {
	    tmpNBDF *= dimensionsForVariables[var];
	}
	tmpNBDF = tmpNBDF - 1;
	diffNbParameters -= tmpNBDF;

	return diffNbParameters;
    }

    /**
     * @return the number of parameters of the models
     */
    public long getNbParameters() {
	if (nbParameters == -1) {
	    List<BitSet> cliques = graph.getCliquesBFS();
	    List<BitSet> separators = graph.getSeparatorsBFS();
	    nbParameters = 0;

	    for (BitSet clique : cliques) {
		int tmpNBDF = 1;
		for (int var = clique.nextSetBit(0); var >= 0; var = clique.nextSetBit(var + 1)) {
		    tmpNBDF *= dimensionsForVariables[var];
		}
		tmpNBDF = tmpNBDF - 1;
		nbParameters += tmpNBDF;
	    }

	    for (BitSet separator : separators) {
		int tmpNBDF = 1;
		for (int var = separator.nextSetBit(0); var >= 0; var = separator.nextSetBit(var + 1)) {
		    tmpNBDF *= dimensionsForVariables[var];
		}
		tmpNBDF = tmpNBDF - 1;
		nbParameters -= tmpNBDF;
	    }

	}

	return nbParameters;

    }

    @Override
    public String toString() {
	List<BitSet> cliques = graph.getCliquesBFS();
	String res = "";
	for (BitSet clique : cliques) {
	    res += "[";
	    for (int var = clique.nextSetBit(0); var >= 0; var = clique.nextSetBit(var + 1)) {
		res += var + " ";
	    }
	    res += "]";
	}
	return res;
    }

    /**
     * @param variableNames
     *            names of the variables
     * @return a string representation of the model
     */
    public String toString(String[] variableNames) {
	List<BitSet> cliques = graph.getCliques();
	String res = "";
	for (BitSet clique : cliques) {
	    res += "[";
	    for (int var = clique.nextSetBit(0); var >= 0; var = clique.nextSetBit(var + 1)) {
		res += variableNames[var] + " ";
	    }
	    res += "]";
	}
	return res;
    }

    /**
     * Export a dot file representing the graph
     * 
     * @see {@linkplain http://www.graphviz.org/content/dot-language}
     * @param file
     *            the file to save the representation to
     * @param variableNames
     *            the names of the variables
     */
    public void exportDOT(File file, String[] variableNames) {
	// String[] simplVar = Arrays.copyOf(variableNames,
	// variableNames.length);
	// for(int i=0;i<simplVar.length;i++){
	// simplVar[i] = simplVar[i].replaceAll("[%()]","");
	// }
	try {
	    PrintWriter out = new PrintWriter(new FileOutputStream(file), true);
	    out.println("graph G{");
	    for (DefaultEdge edge : graph.edgeSet()) {
		if (variableNames == null) {
		    out.println(graph.getEdgeSource(edge) + "--" + graph.getEdgeTarget(edge));
		} else {
		    out.println("\"" + variableNames[graph.getEdgeSource(edge)] + "\"--\"" + variableNames[graph.getEdgeTarget(edge)] + "\"");
		}
	    }
	    out.println("}");

	    out.close();
	} catch (FileNotFoundException e1) {
	    e1.printStackTrace();
	}

    }

    /**
     * Export a JSON file representing the graph
     * 
     * @param file
     *            the file to save the representation to
     * @param variableNames
     *            the names of the variables
     */
    public void exportJSON(File file, String[] variableNames) {
	// String[] simplVar = Arrays.copyOf(variableNames,
	// variableNames.length);
	// for(int i=0;i<simplVar.length;i++){
	// simplVar[i] = simplVar[i].replaceAll("[%()]","");
	// }
	try {
	    PrintWriter out = new PrintWriter(new FileOutputStream(file), true);
	    out.println("{");
	    out.println("\t\"nodes\": [");

	    ArrayList<Integer> listNodes = new ArrayList<Integer>(graph.vertexSet());

	    Integer firstNode = listNodes.get(0);
	    out.println("\t\t{");
	    out.println("\t\t\t\"id\":\"" + firstNode + "\",");
	    if (variableNames != null) {
		out.println("\t\t\t\"text\":\"" + variableNames[firstNode] + "\",");
	    }
	    out.println("\t\t\t\"Type\":\"Symptom\"");
	    out.println("\t\t}");

	    for (int i = 1; i < listNodes.size(); i++) {
		Integer node = listNodes.get(i);
		out.println("\t\t,{");
		out.println("\t\t\t\"id\":\"" + node + "\",");
		if (variableNames != null) {
		    out.println("\t\t\t\"text\":\"" + variableNames[node] + "\",");
		}
		out.println("\t\t\t\"Type\":\"Symptom\"");
		out.println("\t\t}");
	    }

	    out.println("\t],\"links\": [");

	    ArrayList<DefaultEdge> listEdges = new ArrayList<DefaultEdge>(graph.edgeSet());
	    if (!listEdges.isEmpty()) {
		DefaultEdge firstEdge = listEdges.get(0);
		Integer source = graph.getEdgeSource(firstEdge);
		Integer target = graph.getEdgeTarget(firstEdge);
		out.println("\t\t{");
		out.println("\t\t\t\"source\":\"" + source + "\",");
		out.println("\t\t\t\"target\":\"" + target + "\"");
		out.println("\t\t}");

		for (int i = 1; i < listEdges.size(); i++) {
		    DefaultEdge edge = listEdges.get(i);
		    source = graph.getEdgeSource(edge);
		    target = graph.getEdgeTarget(edge);
		    out.println("\t\t,{");
		    out.println("\t\t\t\"source\":\"" + source + "\",");
		    out.println("\t\t\t\"target\":\"" + target + "\"");
		    out.println("\t\t}");
		}
	    }
	    out.println("\t]");
	    out.println("}");

	    out.close();
	} catch (FileNotFoundException e1) {
	    e1.printStackTrace();
	}

    }

    /**
     * Export a dot file representing the graph
     * 
     * @see {@linkplain http://www.graphviz.org/content/dot-language}
     * @param file
     *            the file to save the representation to
     * @param variableNames
     *            the names of the variables
     */
    public void exportDOTCG(File file, String[] variableNames) {
	// String[] simplVar = Arrays.copyOf(variableNames,
	// variableNames.length);
	// for(int i=0;i<simplVar.length;i++){
	// simplVar[i] = simplVar[i].replaceAll("[%()]","");
	// }
	try {
	    PrintWriter out = new PrintWriter(new FileOutputStream(file), true);
	    out.println("graph G{");

	    for (CliqueGraphEdge edge : graph.cg.edgeSet()) {
		BitSet source = graph.cg.getEdgeSource(edge);
		BitSet target = graph.cg.getEdgeTarget(edge);
		BitSet inter = (BitSet) source.clone();
		inter.and(target);
		if (variableNames == null) {
		    out.println(graph.cg.getEdgeSource(edge) + "--" + graph.cg.getEdgeTarget(edge));
		} else {
		    out.print("\"");
		    for (int v = source.nextSetBit(0); v >= 0; v = source.nextSetBit(v + 1)) {
			out.print(variableNames[v] + ";");
		    }
		    out.print("\"--\"");
		    for (int v = target.nextSetBit(0); v >= 0; v = target.nextSetBit(v + 1)) {
			out.print(variableNames[v] + ";");
		    }
		    out.print("\" [label = \"");
		    for (int v = inter.nextSetBit(0); v >= 0; v = inter.nextSetBit(v + 1)) {
			out.print(variableNames[v] + ";");
		    }
		    out.println("\"]");
		}
	    }

	    out.println("}");

	    out.close();
	} catch (FileNotFoundException e1) {
	    e1.printStackTrace();
	}

    }

    /**
     * Returns if the model contains the given interaction
     * 
     * @param v1
     *            first variable
     * @param v2
     *            second variable
     * @return <b>true</b> if the model already considers this interaction
     */
    public boolean containsInteraction(Integer v1, Integer v2) {
	return graph.containsEdge(v1, v2);
    }

    /**
     * Return all the interactions considered by the model
     * 
     * @return all the interactions considered by the model
     */
    public TreeSet<Couple<Integer>> getInteractions() {
	TreeSet<Couple<Integer>> interactions = new TreeSet<Couple<Integer>>();
	for (int v1 = 0; v1 < dimensionsForVariables.length; v1++) {
	    for (int v2 = v1 + 1; v2 < dimensionsForVariables.length; v2++) {
		if (containsInteraction(v1, v2)) {
		    interactions.add(new Couple<Integer>(v1, v2));
		}
	    }
	}
	return interactions;
    }

    public ChordalGraph getGraph() {
	return graph;
    }

    /**
     * @return a Bayesian Network that can represent the same joint (links only,
     *         no CPTs)
     * @throws CycleFoundException
     */
    public DirectedAcyclicGraph<Integer, DefaultEdge> getBayesianNetwork() throws CycleFoundException {
	return graph.getBayesianNetwork();
    }

    public DirectedAcyclicGraph<Integer, DefaultEdge> getBayesianNetwork(int[] preferedOrder) {
	DirectedAcyclicGraph<Integer, DefaultEdge> bn = null;
	try {
	    bn = graph.getBayesianNetwork(preferedOrder);
	} catch (CycleFoundException e) {
	    System.err.println("Found loop in BN");
	    e.printStackTrace();
	}
	return bn;
    }

    public List<BitSet> getCliques() {
	return graph.getCliques();
    }

    public List<BitSet> getCliquesBFS() {
	return graph.getCliquesBFS();
    }

    public List<BitSet> getSeparators() {
	return graph.getSeparatorsBFS();
    }

    public void exportGEXF(File file, String[] variableNames, int[] frequencies, ArrayList<GraphAction> operationsPerformed) {

	// String[] simplVar = Arrays.copyOf(variableNames,
	// variableNames.length);
	// for(int i=0;i<simplVar.length;i++){
	// simplVar[i] = simplVar[i].replaceAll("[%()]","");
	// }
	try {
	    PrintWriter out = new PrintWriter(new FileOutputStream(file), true);

	    out.println("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
	    out.println("<gexf xmlns=\"http://www.gexf.net/1.2draft\" xmlns:viz=\"http://www.gexf.net/1.1draft/viz\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd\" version=\"1.2\">");
	    out.println("\t<graph mode=\"static\" defaultedgetype=\"undirected\">");

	    ArrayList<Integer> listNodes = new ArrayList<Integer>(graph.vertexSet());

	    out.println("\t\t<nodes>");
	    for (int i = 0; i < listNodes.size(); i++) {
		Integer node = listNodes.get(i);
		out.print("\t\t\t<node id=\"" + node + "\"");
		if (variableNames != null) {
		    out.print(" label=\"" + variableNames[node] + "\"");
		}
		out.println(">");
		if (frequencies != null) {
		    out.println("\t\t\t\t<viz:size value=\"" + FastMath.log(frequencies[node]) + "\"/>");
		} else {
		    out.println("\t\t\t\t<viz:size value=\"1\"/>");
		}
		out.print("\t\t\t</node>");
	    }
	    out.println("\t\t</nodes>");
	    out.println("\t\t<edges>");

	    if (!operationsPerformed.isEmpty()) {
		for (int i = 0; i < operationsPerformed.size(); i++) {
		    Double score;
		    GraphAction action = operationsPerformed.get(i);
		    Integer source = action.getV1();
		    Integer target = action.getV2();
		    if (action instanceof PValueScoredGraphAction) {

			score = ((PValueScoredGraphAction) action).entropy;
		    } else {
			score = ((ScoredGraphAction) action).getScore();
		    }
		    out.print("\t\t\t<edge id=\"" + i + "\" source=\"" + source + "\" target=\"" + target + "\" weight=\"" + score + "\" />");
		}
	    }
	    out.println("\t\t</edges>");
	    out.println("\t</graph>");
	    out.println("</gexf>");

	    out.close();
	} catch (FileNotFoundException e1) {
	    e1.printStackTrace();
	}
    }

    public void exportDOT(File file, String[] variableNames, int[] frequencies, ArrayList<GraphAction> operationsPerformed) {

	try {
	    PrintWriter out = new PrintWriter(new FileOutputStream(file), true);

	    out.println("graph Chordalysis{");
	    out.println("\tgraph [K=1.0 fontsize=14 overlap=\"10:\" splines=\"true\"];");
	    out.println("\tratio = auto;");
	    ArrayList<Integer> listNodes = new ArrayList<Integer>(graph.vertexSet());

	    double minFreq = Double.MAX_VALUE, maxFreq = 0.0;
	    if (frequencies != null) {
		// computing the normalisation
		for (int f : frequencies) {
		    double freq = Math.sqrt(f);
		    if (freq < minFreq) {
			minFreq = freq;
		    }
		    if (maxFreq < freq) {
			maxFreq = freq;
		    }
		}
	    }

	    for (int i = 0; i < listNodes.size(); i++) {
		Integer node = listNodes.get(i);
		out.print("\t\"" + node + "\"");
		out.print(" [shape=\"ellipse\" ");
		if (variableNames != null) {
		    out.print("label=\"" + variableNames[node] + "\" ");
		}
		double fontSize;
		if (frequencies != null) {
		    fontSize = 10.0 + (Math.sqrt(frequencies[node]) - minFreq) / (maxFreq - minFreq) * 20;
		} else {
		    fontSize = 10.0;
		}
		out.print("fontsize=" + fontSize + " ");
		out.println("];");
	    }

	    // computing the normalisation
	    double minScore = Double.MAX_VALUE, maxScore = 0.0;
	    if (operationsPerformed != null) {
		for (int i = 0; i < operationsPerformed.size(); i++) {
		    double score;
		    GraphAction action = operationsPerformed.get(i);
		    if (action instanceof PValueScoredGraphAction) {
			score = ((PValueScoredGraphAction) action).entropy;
		    } else {
			score = ((ScoredGraphAction) action).score;
		    }
		    if (score < minScore) {
			minScore = score;
		    }
		    if (maxScore < score) {
			maxScore = score;
		    }
		}
	    }

	    if (!operationsPerformed.isEmpty()) {
		for (int i = 0; i < operationsPerformed.size(); i++) {
		    Double score;
		    GraphAction action = operationsPerformed.get(i);
		    Integer source = action.getV1();
		    Integer target = action.getV2();
		    if (action instanceof PValueScoredGraphAction) {

			score = ((PValueScoredGraphAction) action).entropy;
		    } else {
			score = ((ScoredGraphAction) action).getScore();
		    }
		    double lineWidth = 1.0 + (score - minScore) / (maxScore - minScore) * 10;
		    out.println("\t\"" + source + "\" -- \"" + target + "\" [ penwidth=" + lineWidth + " ];");
		}
	    }
	    out.println("}");

	    out.close();
	} catch (FileNotFoundException e1) {
	    e1.printStackTrace();
	}

    }

    public BitSet findSab(GraphAction action) {
	int a = action.getV1();
	int b = action.getV2();

	BitSet Sab = null;
	for (CliqueGraphEdge e : graph.cg.edgeSet()) {
	    BitSet clique1 = e.getClique1();
	    BitSet clique2 = e.getClique2();
	    if ((clique1.get(a) && clique2.get(b)) || (clique2.get(a) && clique1.get(b))) {
		Sab = e.getSeparator();
		break;
	    }
	}
	if (Sab == null) {// disconnected components
	    Sab = new BitSet(dimensionsForVariables.length);
	}

	if (!Sab.equals(graph.getSeparator(a, b))) {
	    System.err.println("Ouch");
	}
	return Sab;

    }

    public Double entropyDiffIfAdding(int a, int b, BitSet Sab, EntropyComputer computer) {
	BitSet Sabua = (BitSet) Sab.clone();
	BitSet Sabub = (BitSet) Sab.clone();
	BitSet Sabuaub = (BitSet) Sab.clone();
	Sabua.set(a);
	Sabub.set(b);
	Sabuaub.set(a);
	Sabuaub.set(b);

	Double entropy = 0.0;
	Double tmp;

	// Sab
	tmp = computer.computeEntropy(Sab);
	if (tmp == null) {
	    entropy = null;
	    return entropy;
	} else {
	    entropy -= tmp;
	}

	// Sab + a
	tmp = computer.computeEntropy(Sabua);
	if (tmp == null) {
	    entropy = null;
	    return entropy;
	} else {
	    entropy += tmp;
	}

	// Sab + b
	tmp = computer.computeEntropy(Sabub);
	if (tmp == null) {
	    entropy = null;
	    return entropy;
	} else {
	    entropy += tmp;
	}

	// Sab + a + b
	tmp = computer.computeEntropy(Sabuaub);
	if (tmp == null) {
	    entropy = null;
	    return entropy;
	} else {
	    entropy -= tmp;
	}

	return entropy;
    }

    public long nbFreeParametersDiffIfAdding(Integer a, Integer b, NFreeParamsEstimator nFreeParamsEstimator) {
	// System.out.println("computing actual entropy");
	BitSet Sab = graph.getSeparator(a, b);
	BitSet Sabua = (BitSet) Sab.clone();
	BitSet Sabub = (BitSet) Sab.clone();
	BitSet Sabuaub = (BitSet) Sab.clone();
	Sabua.set(a);
	Sabub.set(b);
	Sabuaub.set(a);
	Sabuaub.set(b);

	long diffNbParameters = 0;

	// Sab
	diffNbParameters += nFreeParamsEstimator.computeNFreeParams(Sab);

	// Sab + a
	diffNbParameters -= nFreeParamsEstimator.computeNFreeParams(Sabua);

	// Sab + a + b
	diffNbParameters += nFreeParamsEstimator.computeNFreeParams(Sabuaub);

	// Sab + b
	diffNbParameters -= nFreeParamsEstimator.computeNFreeParams(Sabub);

	return diffNbParameters;

    }

}
