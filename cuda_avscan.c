#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>


#define genedebug 0
#define geneprintf(format, ...) if(genedebug) fprintf(stderr, format, ##__VA_ARGS__)


typedef struct {
    char scan_root[256];
    char sig_file[256];
} cudav_options_t;


void show_usage(int argc, char **argv)
{
    printf("Usage: %s [OPTION]\n", argv[0]);
    printf("  -s, --scan_root=DIR       recursively scan directory. default is \"./\"\n"
           "  -d, --sig_file=FILE       signature file. default \"./generated_sigs.cdv\"\n"
           "\n");
}


int parse_args(cudav_options_t *opts, int argc, char** argv)
{
    /* Set default values */
    memset(opts, 0, sizeof(cudav_options_t));
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


int main(int argc, char **argv)
{
    cudav_options_t opts;
    if (parse_args(&opts, argc, argv) < 0)
        show_usage(argc, argv);

    unsigned int filter_size = 256; /* in bytes */

    /* This is on host, but want on device
    unsigned char *bf = (unsigned char *) malloc(sizeof(unsigned char) * filter_size);
    if (!bf)
        die("malloc failed\n");

    memset(bf, 0, filter_size);

    free(bf);
    */
    return 0;
}
