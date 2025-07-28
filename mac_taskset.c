#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>

void pin_thread_to_core_tag(int tag) {
    thread_affinity_policy_data_t policy = { .affinity_tag = tag };
    thread_act_t thread = mach_thread_self();

    kern_return_t result = thread_policy_set(
        thread,
        THREAD_AFFINITY_POLICY,
        (thread_policy_t)&policy,
        THREAD_AFFINITY_POLICY_COUNT
    );

    if (result != KERN_SUCCESS) {
        // On modern macOS, this might fail if the tag is not what the kernel expects.
        // For Apple Silicon, the physical core number is often the correct tag.
        fprintf(stderr, "mac_taskset: WARNING - Could not set thread affinity tag %d. Error: %d\n", tag, result);
        // We'll proceed with a warning instead of exiting.
    }

    mach_port_deallocate(mach_task_self(), thread);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <core_tag> <command> [args...]\n", argv[0]);
        fprintf(stderr, "Example: %s 6 ./my_benchmark\n", argv[0]);
        return EXIT_FAILURE;
    }

    int core_tag = atoi(argv[1]);
    if (core_tag <= 0) {
        fprintf(stderr, "mac_taskset: ERROR - Core tag must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    pin_thread_to_core_tag(core_tag);
    execvp(argv[2], &argv[2]);

    perror("mac_taskset: execvp failed");
    return EXIT_FAILURE;
}
