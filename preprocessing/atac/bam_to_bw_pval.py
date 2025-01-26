import os
import signal
import subprocess
import time


def bam_to_bw(bam_file, out_prefix, chrom_sizes):
    assert bam_file.endswith('.bam'), f'Input file {bam_file} is not a BAM file'

    _macs2_signal_track(bam_file=bam_file, prefix=out_prefix, chr_sz=chrom_sizes)


def _macs2_signal_track(bam_file, prefix, chr_sz ,gen_sz='hs', pval_thresh=0.01, smooth_win=150):
    # gen_sz - Genome size (sum of entries in 2nd column of chr. sizes file, or hs for human, ms for mouse)
    pval_bigwig = f'{prefix}.pval.signal.bigwig'

    # temporary files
    pval_bedgraph = f'{prefix}.pval.signal.bedgraph'
    pval_bedgraph_srt = f'{prefix}.pval.signal.srt.bedgraph'

    shift_size = -int(round(float(smooth_win) / 2.0))
    temp_files = []
    
    # call peaks using MACS2
    run_shell_cmd(
        f'macs2 callpeak -t {bam_file} -f BAM -n {prefix} -g {gen_sz} -p {pval_thresh} '
        f'--shift {shift_size} --extsize {smooth_win} '
        '--nomodel -B --SPMR --keep-dup all --call-summits '
    )
    
    # sval counts the number of tags per million in the (compressed) BAM file
    sval = _get_bam_lines(bam_file) / 1000000.0
    
    run_shell_cmd(
        f'macs2 bdgcmp -t "{prefix}_treat_pileup.bdg" -c "{prefix}_control_lambda.bdg" '
        f'--o-prefix {prefix} -m ppois -S {sval}')

    # extra step: remove chromosomes not listed from the bedgraph
    run_shell_cmd("awk 'NR==FNR{chromosomes[$1]; next} $1 in chromosomes' " +
                  f'{chr_sz} {prefix}_ppois.bdg > {prefix}_ppois2.bdg')
    
    # bedClip is a utility to trim off any regions in a bed file that are outside the bounds of a specified genome
    run_shell_cmd(f'bedtools slop -i "{prefix}_ppois2.bdg" -g {chr_sz} -b 0 | bedClip stdin {chr_sz} {pval_bedgraph}')
    
    # sort and remove any overlapping regions in bedgraph by comparing two lines in a row
    sort_fn = 'sort'  # 'sort'
    run_shell_cmd(
        f'LC_COLLATE=C {sort_fn} -k1,1 -k2,2n {pval_bedgraph} | '
        f'awk \'BEGIN{{OFS="\\t"}}{{if (NR==1 || NR>1 && (prev_chr!=$1 '
        f'|| prev_chr==$1 && prev_chr_e<=$2)) '
        f'{{print $0}}; prev_chr=$1; prev_chr_e=$3;}}\' > {pval_bedgraph_srt}'
    )
    rm_f(pval_bedgraph)
    
    
    run_shell_cmd(f'bedGraphToBigWig {pval_bedgraph_srt} {chr_sz} {pval_bigwig}')
    rm_f(pval_bedgraph_srt)

    #remove temporary files
    run_shell_cmd(f'mv {prefix}_peaks.narrowPeak {prefix}.peaks.narrowpeak')
    temp_files.append(f'{prefix}_*')
    rm_f(temp_files)
    

def _get_bam_lines(bam_file) -> int:
    return int(run_shell_cmd(f'samtools view -c {bam_file}'))


def _get_ticks():
    """Returns ticks.
        - Python3: Use time.perf_counter().
        - Python2: Use time.time().
    """
    return getattr(time, 'perf_counter', getattr(time, 'time'))()


def run_shell_cmd(cmd):
    p = subprocess.Popen(
        ['/bin/bash', '-o', 'pipefail'],  # to catch error in pipe
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        preexec_fn=os.setsid)  # to make a new process with a new PGID
    pid = p.pid
    pgid = os.getpgid(pid)
    print('run_shell_cmd: PID={}, PGID={}, CMD={}'.format(pid, pgid, cmd))
    t0 = _get_ticks()
    stdout, stderr = p.communicate(cmd)
    rc = p.returncode
    t1 = _get_ticks()
    err_str = f'PID={pid}, PGID={pgid}, RC={rc}, DURATION_SEC={t1 - t0:.1f}\n' \
              f'STDERR={stderr.strip()}\nSTDOUT={stdout.strip()}'

    if rc:
        # kill all child processes
        try:
            os.killpg(pgid, signal.SIGKILL)
        except:
            pass
        finally:
            raise Exception(err_str)
    else:
        print(err_str)
    return stdout.strip('\n')


def rm_f(files):
    if files:
        if type(files) == list:
            run_shell_cmd(f'rm -f {" ".join(files)}')
        else:
            run_shell_cmd(f'rm -f {files}')