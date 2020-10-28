#!/usr/bin/env python
#  coding=utf-8

import argparse
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import math

import utils.log as ul
import utils.chem as uc
import configuration.config_default as cfgd
import utils.file as uf
import utils.plot as up
import configuration.opts as opts
from postprocess import draw_molecules

NUM_WORKERS = 16

class EvaluationRunner:
    """Evaluate the generated molecules"""

    def __init__(self, data_path, num_samples, range_evaluation):

        self.save_path = uf.get_parent_dir(data_path)
        global LOG
        LOG = ul.get_logger(name="evaluation", log_path=os.path.join(self.save_path, 'evaluation.log'))
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path, sep=",")
        self.num_samples = num_samples

        self.output_path = self.save_path
        self.range_evaluation = range_evaluation
        if self.range_evaluation != "":
            self.output_path = os.path.join(self.output_path, '{}'.format(self.range_evaluation))
        uf.make_directory(self.output_path)


    def evaluation_statistics(self):

        # Look at properties separately
        self.property_stat()
        # Satisfying multiple properties
        self.property_overview_stat()

        # Compute Tanimoto similarity
        self.compute_similarity()

        # Save to file
        out_file = self.data_path.split(".csv")[0] + "_statistics.csv"
        self.data.to_csv(out_file, index=False)

        # Draw molecules
        LOG.info("Drawing molecules")
        image = draw_molecules.get_plot_sample(self.data, nr_of_source_mol=50, range_evaluation=self.range_evaluation)
        image.save(os.path.join(self.output_path, 'draw_molecules.png'), format='png')

    def property_stat(self):
        LOG.info("-----------------Looking at properties separately---------------------------")
        for property_name in cfgd.PROPERTIES:
            LOG.info('========{}========'.format(property_name))
            if property_name in ['LogD']:
                self.logD_stat()
            elif property_name in ['Solubility', 'Clint']:
                self.class_prop_stat(property_name)

    def logD_stat(self):
        property_name = "LogD"

        self.logD_stat_prep()

        plus_minus_change = [cfgd.PROPERTY_ERROR['LogD']]
        is_inrange_sum_diffrange = []
        soruce_property = self.data['Source_Mol_{}'.format(property_name)].tolist()
        delta_property = self.data['Delta_{}_ori'.format(property_name)].tolist()
        range_evaluation_pool = [self.range_evaluation] * len(self.data)
        for flu in plus_minus_change:  # for each fluctuation
            range_list = [flu] * len(self.data)
            is_inrange_sum = np.zeros(len(self.data))
            for i in range(self.num_samples):
                pred_logD = self.data['Predict_smi_{}_c{}'.format(i + 1, property_name)].tolist()
                zipped = list(zip(pred_logD, soruce_property, delta_property, range_list, range_evaluation_pool))
                with Pool(NUM_WORKERS) as p:
                    is_inrange = np.asarray(p.map(is_inrange_pool, zipped))
                is_inrange_sum += is_inrange
                self.data['Predict_eval_{}_{}_{}'.format(i + 1, property_name, flu)] = is_inrange
            is_inrange_sum_diffrange.append(is_inrange_sum)

        LOG.info("Percentage of test molecules that have at least 1 generated molecule "
                 "falls in the range of desirable delta_{}+-: ".format(property_name))
        for i, flu in enumerate(plus_minus_change):
            label = 'num_in_accepted_range_{}_{}'.format(flu, property_name)
            self.data[label] = is_inrange_sum_diffrange[i]  # [count among num_samples]
            up.hist(self.data, label, name=label, path=self.output_path,
                    title=r"Number of molecules with LogD in desired$\pm${}".format(flu))

            # best prediction
            temp_df = self.data[abs(self.data['Delta_{}_predict_best'.format(property_name)] -
                                    self.data['Delta_{}_ori'.format(property_name)]) <= flu]
            LOG.info("{}    {:.2f}%".format(flu, len(temp_df) * 100.0 / len(self.data)))

        LOG.info(
            "Median number of molecules among {} sampled in desirable {}+-".format(self.num_samples, property_name))
        for i, flu in enumerate(plus_minus_change):
            label = 'num_in_accepted_range_{}_{}'.format(flu, property_name)

            t = self.data[label].tolist()
            t_median = np.median(t)
            LOG.info("{}: {} ".format(flu, t_median))

        for i, flu in enumerate(plus_minus_change):
            label = 'num_in_accepted_range_{}_{}'.format(flu, property_name)

            up.hist_box(self.data, label, name=label + "_hist_box",
                        path=self.output_path,
                        title=r"Number of molecules with desired LogD $\pm${}".format(flu))
            LOG.info(self.data[label].describe())


    def logD_stat_prep(self):
        def delta_value_close_to_desired(delta_x, delta_y, delta_true):
            if abs(delta_x - delta_true) < abs(delta_y - delta_true):
                return delta_x
            else:
                return delta_y

        property_name = "LogD"
        if self.data_path == 'test_unseen_L-1_S01_C10_range':
            self.data['Delta_{}_ori'.format(property_name)] = -1
        else:
            self.data['Delta_{}_ori'.format(property_name)] = self.data['Target_Mol_{}'.format(property_name)] - \
                                                          self.data['Source_Mol_{}'.format(property_name)]

        best_prediction = [float("inf")] * len(self.data)  # predicted_property - source_property
        for i in range(self.num_samples):
            # Delta_LogD predicted
            self.data['Delta_{}_predict_{}'.format(property_name, i + 1)] = self.data['Predict_smi_{}_c{}'.format(
                i + 1, property_name)] - self.data['Source_Mol_{}'.format(property_name)]
            self.data['Delta_{}_predict_{}'.format(property_name, i + 1)] = self.data[
                'Delta_{}_predict_{}'.format(property_name, i + 1)].astype(float)

            delta_x = self.data['Delta_{}_predict_{}'.format(property_name, i + 1)].tolist()
            delta_true = self.data['Delta_{}_ori'.format(property_name)].tolist()
            best_prediction = list(map(delta_value_close_to_desired, delta_x, best_prediction, delta_true))

        # best delta logD from generated molecules
        self.data['Delta_{}_predict_best'.format(property_name)] = best_prediction
        self.data['Delta_{}_predict_best'.format(property_name)] = self.data[
            'Delta_{}_predict_best'.format(property_name)].astype(float)

    def compute_similarity(self):
        LOG.info('Computing Tanimoto similarity')
        source_smiles_list = self.data['Source_Mol'].tolist()
        similarities = []
        for i in range(self.num_samples):
            pred_smi_list = self.data['Predicted_smi_{}'.format(i + 1)].tolist()
            zipped = list(zip(source_smiles_list, pred_smi_list))
            with Pool(NUM_WORKERS) as p:
                results = p.map(uc.tanimoto_similarity_pool, zipped)
            similarities.extend(results)
        results_not_none = [s for s in similarities if s]
        up.hist_box_list(results_not_none, name="similarity",
                    path=self.output_path, title="Similarity")


    def class_prop_stat(self, property_name):
        threshold = cfgd.PROPERTY_THRESHOLD[property_name]

        # --------------Solubility and Clint class for source molecule
        pred_prop = self.data['Source_Mol_{}'.format(property_name)].tolist()
        threshold_pool = [threshold] * len(self.data)
        error_pool = [0]*len(self.data)
        zipped = list(zip(pred_prop, threshold_pool, error_pool))
        with Pool(NUM_WORKERS) as p:
            class_label_list = np.asarray(p.map(class_label_pool, zipped))
        self.data['Source_Mol_{}_class'.format(property_name)] = class_label_list

        # Generated molecules
        error = cfgd.PROPERTY_ERROR[property_name]
        plus_minus_change = [error]
        delta_property = self.data['Delta_{}'.format(property_name)].tolist()

        for flu in plus_minus_change:  # for each fluctuation
            is_inrange_sum = np.zeros(len(self.data))
            for i in range(self.num_samples):
                pred_prop = self.data[f'Predict_smi_{i + 1}_c{property_name}'].tolist()
                flu_pool = [flu]*len(self.data)
                zipped = list(zip(pred_prop, threshold_pool, flu_pool))
                with Pool(NUM_WORKERS) as p:
                    class_label_list = np.asarray(p.map(class_label_pool, zipped))
                self.data[f'Predict_smi_{i+1}_c{property_name}_{flu}_class'] = class_label_list

                source_class_pool = self.data['Source_Mol_{}_class'.format(property_name)].tolist()
                pred_class_pool = self.data[f'Predict_smi_{i+1}_c{property_name}_{flu}_class'].tolist()
                zipped = list(zip(source_class_pool, pred_class_pool))
                with Pool(NUM_WORKERS) as p:
                    prop_change_list = np.asarray(p.map(prop_change_pool, zipped))
                self.data[f'Delta_{property_name}_predict_{i+1}_{flu}'] = prop_change_list

                delta_property_p = self.data[f'Delta_{property_name}_predict_{i+1}_{flu}'].tolist()
                zipped = list(zip(delta_property, delta_property_p))
                with Pool(NUM_WORKERS) as p:
                    is_inrange = np.asarray(p.map(is_inrange_class_pool, zipped))
                is_inrange_sum += is_inrange
                self.data['Predict_eval_{}_{}_{}'.format(i + 1, property_name, flu)] = is_inrange

            label = 'num_correct_{}_{}'.format(property_name, flu)
            self.data[label] = is_inrange_sum  # [count among num_samples]

            LOG.info('Percentage of test molecules that have at least 1 generated molecule satisfying '
                     'desirable change+-{} : {:.2f}%'.format(flu,
                                               np.count_nonzero(is_inrange_sum) * 1.0 / len(is_inrange_sum)*100))
            t = self.data[label].tolist()
            t_median = np.median(t)
            LOG.info("Median number of molecules among {} sampled "
                     "that satisfy desirable change+-{}:  {}".format(self.num_samples, flu, t_median))

        for flu in plus_minus_change:  # for each fluctuation
            label = 'num_correct_{}_{}'.format(property_name, flu)
            up.hist(self.data, label, name=label, path=self.output_path,
                    title=r"Number of molecules with desired {} $\pm${}".format(property_name, flu))

            up.hist_box(self.data, label, name=label + "_hist_box",
                        path=self.output_path,
                        title=r"Number of molecules with desired {} $\pm${}".format(property_name, flu))
            LOG.info(self.data[label].describe())

    def property_overview_stat(self):
        LOG.info('---------------------Looking at All properties:--------------------------')
        is_inrange_all_property_sumoversample = np.zeros(len(self.data))
        for i in range(self.num_samples):
            # for each sample i
            is_inrange_all_property = np.array([1] * len(self.data))
            # look at all properties
            for property_name in cfgd.PROPERTIES:
                is_inrange_all_property = is_inrange_all_property & self.data[
                    'Predict_eval_{}_{}_{}'.format(i + 1, property_name, cfgd.PROPERTY_ERROR[property_name])]
            self.data['Predict_eval_{}_allprop'.format(i + 1)] = is_inrange_all_property  # [0,1]
            is_inrange_all_property_sumoversample += is_inrange_all_property
        self.data['num_correct_allprop_sumoversample_allerror'] = is_inrange_all_property_sumoversample  # count

        label = 'num_correct_allprop_sumoversample_allerror'
        up.hist(self.data, label, name=label, path=self.output_path,
                title="Number of molecules with desired properties")

        LOG.info("Evaluate the best out of {} sampled, percentage of test molecules that have at least 1 generated molecule "
                 "satisfying all properties: {:.2f}%".format(self.num_samples, np.count_nonzero(is_inrange_all_property_sumoversample) * 100.0 / len(
                is_inrange_all_property_sumoversample)))

        t = self.data[label].tolist()
        t_median = np.median(t)
        LOG.info("Median number of molecules among {} sampled satisfying all properties, {}".format(self.num_samples, t_median))

        up.hist_box(self.data, label, name=label+"_hist_box",
                    path=self.output_path, title="Number of molecules with desired properties")
        LOG.info(self.data[label].describe())

def is_inrange_pool(args):
    return is_inrange(*args)


def is_inrange(pred_logD, source_logD, delta_logD, range, range_evaluation):
    if range_evaluation == "":
        return 1 if abs(pred_logD - (source_logD + delta_logD)) < range else 0
    elif range_evaluation == "lower":
        return 1 if pred_logD >= cfgd.LOD_MIN and pred_logD < min(source_logD + range, cfgd.LOD_MAX) else 0
    elif range_evaluation == "higher":
        return 1 if pred_logD > max(source_logD - range, cfgd.LOD_MIN) and pred_logD <= cfgd.LOD_MAX else 0
    else:
        print("No legal range requirement provided")


def is_inrange_class_pool(args):
    return is_inrange_class(*args)

def is_inrange_class(delta, delta_p):
    if not delta_p:
        print('is_inrange_class ', delta, delta_p)

    return 1 if delta in delta_p else 0

def class_label_pool(args):
    return class_label(*args)

def class_label(prop_value, threshold, error=0):
    if math.isnan(prop_value):
        return 'not_valid'
    if error == 0:
        return 'high' if prop_value > threshold else 'low'
    else:
        if abs(prop_value-threshold) <= error:
            return 'low, high'
        elif prop_value > threshold + error:
            return 'high'
        elif prop_value < threshold - error:
            return 'low'

def prop_change_pool(args):
    return prop_change(*args)

def prop_change(source, target):
    if source == "low":
        if "low" in target and "high" in target:
            return "low->high, no_change"
        elif target == "low":
            return "no_change"
        elif target == "high":
            return "low->high"
        elif target == "not_valid":
            return "low->nan"
    elif source == "high":
        if "low" in target and "high" in target:
            return "high->low, no_change"
        elif target == "low":
            return "high->low"
        elif target == "high":
            return "no_change"
        elif target == "not_valid":
            return "high->nan"

def run_main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='evaluation.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.evaluation_opts(parser)
    opt = parser.parse_args()

    runner = EvaluationRunner(opt.data_path, opt.num_samples, opt.range_evaluation)
    runner.evaluation_statistics()

if __name__ == "__main__":
    run_main()
