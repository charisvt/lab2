#define lm 4
#define ln 4
#define lp 4

#define m (1 << lm)
#define n (1 << ln)
#define p (1 << lp)

typedef unsigned int input_type;
typedef unsigned int result_type;

extern "C" {
void mult_hw(input_type in1[n * m], input_type in2[m * p], result_type out_r[n * p]){
#pragma HLS INTERFACE m_axi bundle=gmem0 port = in1 depth = (n * m)
#pragma HLS INTERFACE m_axi bundle=gmem port = in2 depth = (m * p)
#pragma HLS INTERFACE m_axi bundle=gmem0 port = out_r depth = (n * p)
#pragma HLS INTERFACE s_axilite port = return bundle = control
// we use different memory banks to maximize bandwidth

    input_type A[n * m];
    input_type B[m * p];
    result_type C[n * p];

// use cyclic partition for A as we want row-wise access
#pragma HLS ARRAY_PARTITION variable = A dim = 1 cyclic factor = 16

// use block partition for B as we want column-wise access
#pragma HLS ARRAY_PARTITION variable = B dim = 1 block factor = 16

    // burst read A
    readA:
    	for(int itr=0, i=0, j=0; itr<n*m; itr++,j++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = n * m max = n * m
    		if(j == m){
    			j=0;
    			i++;
    		}
    		A[i * m + j] = in1[itr];
    	}

    // burst read B
    readB:
		for(int i=0; i<m * p; i++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = m * p max = m * p
//#pragma HLS UNROLL factor = 64
			B[i] = in2[i];
    	}

    // calc C
    mult_outer:
    	for(int i=0; i<n; i++){
#pragma HLS LOOP_TRIPCOUNT min = n max = n
    		mult_middle:
    			for(int j=0; j<p; j++){
#pragma HLS LOOP_TRIPCOUNT min = p max = p
    				result_type result = 0;
    				mult_inner:
    				for(int k=0; k<m; k++){
#pragma HLS UNROLL factor=16
    					result += A[i * m + k] * B[k * p + j];
    				}
    				C[i * p + j] = result;
    			}
    	}

    // copy C back to host
    writeC:
    	for(int i=0; i<n*p;i++){
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min = n * p max = n * p
    		out_r[i] = C[i];
    	}
}
}
