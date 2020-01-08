__global__
void total_energy(int N_el, int *el_ind, int *dop_ind, float *site_pos, float *Etot) {
	int i = threadIdx.x;
	int j = blockIdx.x;

	float dist;
	float Etot_local = 0.;

	if ((i < N_el) && (j < N_el)) {
    	if (i != j) {
			int site1, site2;

	    	site1 = el_ind[i] * 2;
	    	site2 = el_ind[j] * 2;
	    	dist = sqrt((site_pos[site1] - site_pos[site2]) * (site_pos[site1] - site_pos[site2])
	    			  + (site_pos[site1 + 1] - site_pos[site2 + 1]) * (site_pos[site1 + 1] - site_pos[site2 + 1]));

	    	Etot_local += 1.0f / dist / 0.69508f / 2.0f;  // /2.0 to balance for double-counting (i, j) equiv. (j, i)
		}
	}

	if ((el_ind[i] != dop_ind[j]) && (i < N_el) && (j < N_el)) {
			int site_el, site_dop;

	    	site_el = el_ind[i] * 2;
	    	site_dop = dop_ind[j] * 2;
	    	dist = sqrt((site_pos[site_el] - site_pos[site_dop]) * (site_pos[site_el] - site_pos[site_dop])
	    			  + (site_pos[site_el + 1] - site_pos[site_dop + 1]) * (site_pos[site_el + 1] - site_pos[site_dop + 1]));

	    	Etot_local += - 1.0f / dist / 0.69508f; // double counting is not an issue here since electrons and dopants are different
	}

	atomicAdd(Etot, Etot_local);
}