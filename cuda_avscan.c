#include <getopt.h>
#include <limits.h> /* for CHAR_BIT */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define genedebug 0
#define geneprintf(format, ...) if(genedebug) fprintf(stderr, format, ##__VA_ARGS__)
#define GENE_HAS_CUDA 0


/* According to "Creating Signatures for ClamAV"
 * <www.clamav.com/doc/latest/signatures.pdf> pg.4:
 * "the recommended size of a hex signature is 40 up to 300 characters,"
 * implying a 20B to 150B signature length.  There are signature lengths up to
 * and greater than 980B:
 * All ClamAV releases older than 0.95 are affected by a bug in freshclam which
 * prevents incremental updates from working with signatures longer than 980B.
 * see: <https://wwws.clamav.net/bugzilla/show_bug.cgi?id=1395>
 *
 * We assume 256B is the signature length.
 */
//#define SIG_LENGTH	256  /* Moved to main() */
//#define SIG_CACHE_SIZE	(SIG_LENGTH * 512)  /* In bytes */

//#define DATA_XFER_QUEUE_LEN 4
//static int data_xfer_queue[DATA_XFER_QUEUE_LEN];

#define BLOOM_FILTER_BITS_PER_SIG 10

struct cuda_avscan_options {
    char scan_root[256];
    char sig_file[256];
} cudav_options_t;


void show_usage(int argc, char **argv)
{
    printf("Usage: %s [OPTION]\n", argv[0]);
    printf("  -s, --scan_root=DIR       recursively scan directory. Default \"./\"\n"
           "  -d, --sig_file=FILE       signature file. Default \"./generated_sigs.cdv\"\n"
           "\n");
}


int parse_args(struct cuda_avscan_options *opts, int argc, char** argv)
{
    /* Set default values */
    memset(opts, 0, sizeof(struct cuda_avscan_options));
    strcpy(opts->scan_root, "./");
    strcpy(opts->sig_file, "./generated_sigs.cdv");

    /* Parse for command line arguments */
    struct option long_options[] = {
     /* name,           has_arg,    flag,       val */
      { "scan_root",    1,          NULL,       0 },
      { "sig_file",     1,          NULL,       0 },
      { "help",         0,          NULL,       0 },
      { NULL,           0,          NULL,       0 },
    };

    int c;
    int option_index = 0;
    for (;;) {
        c = getopt_long(argc, argv, "sd:h", long_options, &option_index);

        /* Detect the end of options */
        if (c == -1)
            break;

        switch (c) {
            case 0:
                if (long_options[option_index].flag != 0)
                    break;

                switch (option_index) {
                    case 0:
                        if (optarg)
                            strcpy(opts->scan_root, optarg);
                    case 1:
                        if (optarg)
                            strcpy(opts->sig_file, optarg);
                    case 2: /* --help */
                        geneprintf("hello\n");
                        show_usage(argc, argv);
                        return 0;
                }
                break;
            case 's':
                strcpy(opts->scan_root, optarg);
                break;
            case 'd':
                strcpy(opts->sig_file, optarg);
                break;
            case 'h':
                show_usage(argc, argv);
            case '?':
                break;
            default:
                return -1;
        }
    }

    return 0;
}


unsigned char *sig_cache_alloc(size_t size_in_bytes)
{
    unsigned char *d_sig_cache = NULL;

#if GENE_HAS_CUDA
    cutilSafeCall (cudaMalloc (d_sig_cache, size_in_bytes));
#endif

    return d_sig_cache;
}


unsigned char *bf_alloc(size_t size_in_bytes)
{
    unsigned char *d_bf = NULL;

#if GENE_HAS_CUDA
    cutilSafeCall (cudaMalloc (d_bf, size_in_bytes));
#endif

    return d_bf;
}



FILE *sig_file_open(const char *sig_file, size_t *num_sigs)
{
    FILE *sig_file_handle = NULL;
    sig_file_handle = fopen(sig_file, "r");

    if (!sig_file_handle)
        return NULL;

    /* Generated signature file prepended with 'int num_sigs' */
    //fseek(sig_file_handle, sizeof (int), SEEK_SET);
    fread(num_sigs, sizeof *(num_sigs), 1, sig_file_handle);

    return sig_file_handle;
}


size_t sig_cache_fill(unsigned char *d_sig_cache, size_t bytes, FILE *sig_file)
{
    char *buf = (char *) malloc(bytes);
    if (!buf)
        return 0;

    size_t r = fread(buf, bytes, 1, sig_file);
    /* TODO: Copy from host to device memory */
    if (r != 0) {
#if GENE_HAS_CUDA
	//cutilSafeCall (cudaStreamCreate (&(sc->h_blocks[index].sid));
        cutilSafeCall (cudaMemcpy (d_sig_cache, buf, bytes, cudaMemcpyHostToDevice));
#if 0
int sig_cache_insert(Signature_Cache* sc,unsigned char* sig_block,
                     int block_size,int index,int sig_count){

	// Create a new stream for this scan kernel
	cudaError_t c = cudaStreamCreate(&(sc->h_blocks[index].sid));

	sc->h_blocks[index].num_sigs = sig_count;

	// copy host memory to device
	if(sc->d_sig_cache == NULL || sig_block == NULL){
	  printf("\nSHIT!\n");
	  exit(1);
	}
        //poss XXX: sig_block isn't page locked? doesn't it have to be allocated using cudaMemAlloc? maybe not, not sure
	cutilSafeCall( cudaMemcpyAsync(sc->d_sig_cache,sig_block,SIG_BLOCK_SIZE,
											 cudaMemcpyHostToDevice,
											 sc->h_blocks[index].sid));
	return index;
}
#endif
#endif
    }

    return r;
}


int main(int argc, char **argv)
{
    /* Leave out for now, can refactor later
    struct gpu_handle {
    } gpu_handle;
    */

    struct cuda_avscan_options opts;
    if (parse_args(&opts, argc, argv) < 0)
        show_usage(argc, argv);


    unsigned char *d_sig_cache;
    const size_t cache_num_sigs = 1000;
    const size_t sig_length = 256; /* See note for SIG_LENGTH */
    size_t sig_cache_bytes = cache_num_sigs * sig_length;

    d_sig_cache = sig_cache_alloc(sig_cache_bytes);


    FILE *sig_file_handle;
    /* For reference:
     *   Latest ClamAV® stable release is: 0.95.3
     *   Total number of signatures: 719461
     */
    size_t total_num_sigs = 0;
    sig_file_handle = sig_file_open(opts.sig_file, &total_num_sigs);

    if (!sig_file_handle) {
        fprintf(stderr, "Cannot open signature file\n");
        return -1;
    }
    if (total_num_sigs <= 0) {
        fprintf(stderr, "Number of signatures not positive\n");
        return -1;
    }


    unsigned char *d_bloom_filter;
    size_t filter_size = total_num_sigs * BLOOM_FILTER_BITS_PER_SIG / CHAR_BIT;
    d_bloom_filter = bf_alloc(filter_size);


    size_t num_sigs_filled;
    num_sigs_filled = sig_cache_fill(d_sig_cache, sig_cache_bytes, sig_file_handle);
    while (num_sigs_filled > 0) {
        int r = -1;
        //r = launch_bf_sig_insert(d_sig_cache, num_sigs_filled, d_bloom_filter);
        if (r < 0) {
            fprintf(stderr,"Failed to insert signature into bloom filter.\n");
            return -1;
        }

        num_sigs_filled = sig_cache_fill(d_sig_cache, sig_cache_bytes, sig_file_handle);
    }

#if 0
    /* This is on host, but want on device
    unsigned char *bf = (unsigned char *) malloc(sizeof(unsigned char) * filter_size);
    if (!bf)
        die("malloc failed\n");

    memset(bf, 0, filter_size);

    free(bf);
    */
#endif
    return 0;
}
