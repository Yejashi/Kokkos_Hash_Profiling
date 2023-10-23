#include <iostream>
#include <cstdio>
#include <Kokkos_Core.hpp>
#include <kokkos_murmur3.hpp>
#include <map_helpers.hpp>


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

    - Deletion Performance
        Time required to delete elements from the table.
        .erase

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
*/
int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        DigestNodeIDDeviceMap device_hash;
        device_hash.rehash(10000);


















        Kokkos::View<NodeID*> sample_data("sample_data", 10000);

        //Create 10,000 elements
        Kokkos::parallel_for("hash_insert", sample_data.extent(0), KOKKOS_LAMBDA(const int i) {
            sample_data(i) = NodeID(2 + i * 12, 3 + i * 7);
        });
        Kokkos::fence();

        Kokkos::parallel_for("hash_insert", 129, KOKKOS_LAMBDA(const int i) {
            HashDigest digest;
            hash(&(sample_data(i)), sizeof(sample_data(i)), digest.digest);
            device_hash.insert(digest, NodeID(7, 1));
        });
        Kokkos::fence();

        Kokkos::parallel_for("hash_insert", 1, KOKKOS_LAMBDA(const int i) {
            auto capacity = device_hash.capacity(); //Device function
            // auto size = device_hash.size();
            printf("Capacity: %d\n", capacity);
            // printf("Size: %d\n", size);
        });

        auto size = device_hash.size();
        printf("Size: %d\n", size);


    }
    Kokkos::finalize();
    return 0;
}