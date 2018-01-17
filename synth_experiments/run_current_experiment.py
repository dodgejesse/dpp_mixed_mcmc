import discrepancy
import compute_discrep_from_saved_data as comp
import center_origin_plots



if __name__ == "__main__":
    discrepancy.draw_many_samples()
    comp.compute_discrep_for_samples()
    center_origin_plots.make_plots()
