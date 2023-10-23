#include <iostream>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <kokkos_murmur3.hpp>
#include <map_helpers.hpp>
#include <math.h>
#include <vector>

/* Notes
    template<class Value, class ExecSpace>
                                     Key         Value             HashFunc     CompFunc
    DigestMap = Kokkos::UnorderedMap<HashDigest, Value, ExecSpace, digest_hash, digest_equal_to>;
    DigestNodeIDDeviceMap = DigestMap<NodeID, Kokkos::DefaultExecutionSpace>;
    
    //VALUE being stored in the hash map
    struct NodeID {
    uint32_t node;
    uint32_t tree;
    
    KOKKOS_INLINE_FUNCTION
    NodeID() {
        node = UINT_MAX;
        tree = UINT_MAX;
    }
    
    KOKKOS_INLINE_FUNCTION
    NodeID(uint32_t n, uint32_t t) {
        node = n;
        tree = t;
    }

    KOKKOS_INLINE_FUNCTION
    bool operator==(const NodeID& other) const {
        return !(other.node != node || other.tree != tree);
    }
    };

    //Insertion
    DigestNodeIDDeviceMap device_hash;
    device_hash.insert([HashDigest], [NodeID)])

    //Sample parallel_for
    Kokkos::parallel_for("hash_insert", 1, KOKKOS_LAMBDA(const int x) {

    //Sample usecase
    HashDigest digest;
    DigestNodeIDDeviceMap device_hash;
    uint32_t sample[5] = {1,2,3,4,5};

    hash(&sample, sizeof(sample), digest.digest);

    Kokkos::parallel_for("hash_insert", 1, KOKKOS_LAMBDA(const int x) {
        device_hash.insert(digest, NodeID(7, 1));
    });

    Kokkos::parallel_for("hash_insert", 1, KOKKOS_LAMBDA(const int x) {
        auto res = device_hash.find(digest);
        printf("%d\n", device_hash.value_at(res).node);
    });

*/

/* Profiling Approach
    - Insertion Performance
        Time it takes to insert elements into the table
        .insert
    
    - Find Performance
        Time it takes to lookup elements in the hash table
        .exists
        .find


    - Collisions and Load Factor
        Rate of collisions along with load factor (stored/capacity)
        .capacity //device function
        .size     //host function

    - Scaling
        How does it perform with different number of threads.

    Sample test
        Given load factor of n.
        Given number of concurrent operations z.
            Insertion Perf
            Find Perf
            Deletion Perf

        15 iterations
        10 20 30 40 50 60 70 80 90 

    v100 5120
    so 5120 insertions per kernel
*/

void create_sample_data(Kokkos::View<uint32_t*> sample_data, Kokkos::View<HashDigest*> sample_digests) {
    Kokkos::parallel_for("hash_insert", sample_data.extent(0), KOKKOS_LAMBDA(const int i) {
        // sample_data(i) = NodeID(2 + i * 12, 3 + i * 7);
        sample_data(i) = i;
        HashDigest digest;
        hash(&(sample_data(i)), sizeof(sample_data(i)), digest.digest);
        sample_digests(i) = digest;
    });
    Kokkos::fence();
}

void fill_until(DigestNodeIDDeviceMap device_hash, Kokkos::View<uint32_t*> sample_data, Kokkos::View<HashDigest*> sample_digests, int fill_size) {
    // int fill_size = (percent_full * hash_capacity) / 100;

    // printf("Hash Size: %d\n", hash_size);
    // printf("Filling up hash table until %d%% capacity.\n", percent_full);
    // printf("Final capacity = %d\n\n", capacity);
    // printf("Initial size %d\n", device_hash.size());

    //This need serious reevaluation -- speak with Nigel
    auto policy = Kokkos::RangePolicy<>(0, fill_size);
    Kokkos::parallel_for("hash_fill", policy, KOKKOS_LAMBDA(const int i) {
        // HashDigest digest;
        // hash(&(sample_data(i)), sizeof(sample_data(i)), digest.digest);
        HashDigest digest = sample_digests(i);
        device_hash.insert(digest, NodeID(sample_data(i), 1));
    });
    Kokkos::fence();
    // printf("Final size %d\n", device_hash.size());

}

void insertion_test(DigestNodeIDDeviceMap device_hash, Kokkos::View<uint32_t*> sample_data, Kokkos::View<HashDigest*> sample_digests, int starting_index, int num_insertions, int capacity, int percent_full) {
    if(num_insertions < 5120) {
        num_insertions = 5120;
    }

    std::string label = "Insertion Test -- Capacity = " + std::to_string(capacity)
    + " -- Percent Full = " + std::to_string(percent_full) + "%";

    auto policy = Kokkos::RangePolicy<>(starting_index, num_insertions);
    Kokkos::parallel_for(label, policy, KOKKOS_LAMBDA(const int i) {
        // HashDigest digest;
        // hash(&(sample_data(i)), sizeof(sample_data(i)), digest.digest);
        HashDigest digest = sample_digests(i);
        device_hash.insert(digest, NodeID(sample_data(i), 1));
    });
    Kokkos::fence();
}

void find_test(DigestNodeIDDeviceMap device_hash, Kokkos::View<uint32_t*> sample_data, Kokkos::View<HashDigest*> sample_digests, int starting_index, int num_finds, int capacity, int percent_full) {
    if(num_finds < 5120) {
        num_finds = 5120;
    }
    
    std::string label = "Find Test -- Capacity = " + std::to_string(capacity)
    + " -- Percent Full = " + std::to_string(percent_full) + "%";

    auto policy = Kokkos::RangePolicy<>(starting_index, num_finds);
    Kokkos::parallel_for(label, policy, KOKKOS_LAMBDA(const int i) {
        // HashDigest digest;
        // hash(&(sample_data(i)), sizeof(sample_data(i)), digest.digest);
        device_hash.find(sample_digests(i));
    });
    Kokkos::fence();
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        int capacity = 10000;
        // Kokkos::View<NodeID*> sample_data("sample_data", capacity * pow(2, 15));
        Kokkos::View<uint32_t*> sample_data("sample_data", capacity * pow(2, 15));
        Kokkos::View<HashDigest*> sample_digests("sample_data", capacity * pow(2, 15));


        // printf("Size of data %d\n", sample_data.extent(0));
        capacity *= 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2;
        create_sample_data(sample_data,sample_digests);

        for(int i = 8; i < 9; ++i) {
            // printf("Current hash size %d\n\n", size);

            //Create a new hash
            DigestNodeIDDeviceMap device_hash;
            device_hash.rehash(capacity);
            device_hash.clear(); //redundant
            printf("Capacity %d\n", capacity);

            //Test for different initial sizes
            for (int j = 0; j < 9; ++j) {
                int percent_full = 10 * (j + 1);
                // int num_insertions = capacity * 0.1;
                int num_insertions = 10000;
                int fill_size = (percent_full * capacity) / 100;

                fill_until(device_hash, sample_data, sample_digests, fill_size);
                insertion_test(device_hash, sample_data, sample_digests, fill_size, num_insertions, capacity, percent_full);
                find_test(device_hash, sample_data, sample_digests,  fill_size, num_insertions, capacity, percent_full);


                device_hash.clear();
            }


            // capacity *= 2;
        }

















        // Kokkos::View<NodeID*> sample_data("sample_data", 10000);

        // //Create 10,000 elements
        // Kokkos::parallel_for("hash_insert", sample_data.extent(0), KOKKOS_LAMBDA(const int i) {
        //     sample_data(i) = NodeID(2 + i * 12, 3 + i * 7);
        // });
        // Kokkos::fence();

        // Kokkos::parallel_for("hash_insert", 129, KOKKOS_LAMBDA(const int i) {
        //     HashDigest digest;
        //     hash(&(sample_data(i)), sizeof(sample_data(i)), digest.digest);
        //     device_hash.insert(digest, NodeID(7, 1));
        // });
        // Kokkos::fence();

        // Kokkos::parallel_for("hash_insert", 1, KOKKOS_LAMBDA(const int i) {
        //     auto capacity = device_hash.capacity(); //Device function
        //     // auto size = device_hash.size();
        //     printf("Capacity: %d\n", capacity);
        //     // printf("Size: %d\n", size);
        // });

        // auto size = device_hash.size();
        // printf("Size: %d\n", size);


    }
    Kokkos::finalize();
    return 0;
}