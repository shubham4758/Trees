1.// Solution

import java.util.ArrayList;
import java.util.List;

public class KClosestElements {
    public static List<Integer> findClosestElements(int[] arr, int k, int x) {
        List<Integer> result = new ArrayList<>();

        // Find the position of the closest element to x using binary search
        int closestIndex = binarySearchClosest(arr, x);

        // Initialize two pointers to expand around the closest element
        int left = closestIndex - 1;
        int right = closestIndex;

        // Expand the range to find the k closest elements
        while (k > 0) {
            if (left < 0 || (right < arr.length && Math.abs(arr[left] - x) > Math.abs(arr[right] - x))) {
                right++;
            } else {
                left--;
            }
            k--;
        }

        // Add the k closest elements to the result list
        for (int i = left + 1; i < right; i++) {
            result.add(arr[i]);
        }

        return result;
    }

    // Binary search to find the position of the closest element to x
    private static int binarySearchClosest(int[] arr, int x) {
        int left = 0;
        int right = arr.length - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;

            if (arr[mid] == x) {
                return mid;
            } else if (arr[mid] < x) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // If there is no exact match, return the index of the closest element
        if (left == 0) {
            return 0;
        } else if (left == arr.length) {
            return arr.length - 1;
        } else {
            return Math.abs(arr[left] - x) < Math.abs(arr[left - 1] - x) ? left : left - 1;
        }
    }

    public static void main(String[] args) {
        int[] arr1 = {1, 2, 3, 4, 5};
        int k1 = 4;
        int x1 = 3;
        List<Integer> result1 = findClosestElements(arr1, k1, x1);
        System.out.println(result1); // Output: [1, 2, 3, 4]

        int[] arr2 = {1, 2, 3, 4, 5};
        int k2 = 4;
        int x2 = -1;
        List<Integer> result2 = findClosestElements(arr2, k2, x2);
        System.out.println(result2); // Output: [1, 2, 3, 4]
    }
}



2.// Solution


public class KthSmallestInSortedMatrix {
    public static int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        int left = matrix[0][0];
        int right = matrix[n - 1][n - 1];

        while (left < right) {
            int mid = left + (right - left) / 2;
            int count = countLessEqual(matrix, mid, n);

            if (count < k) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left;
    }

    private static int countLessEqual(int[][] matrix, int target, int n) {
        int count = 0;
        int row = n - 1;
        int col = 0;

        while (row >= 0 && col < n) {
            if (matrix[row][col] <= target) {
                count += row + 1; // Count elements in the current column up to the target
                col++;
            } else {
                row--;
            }
        }

        return count;
    }

    public static void main(String[] args) {
        int[][] matrix1 = {
            {1, 5, 9},
            {10, 11, 13},
            {12, 13, 15}
        };
        int k1 = 8;
        System.out.println(kthSmallest(matrix1, k1)); // Output: 13

        int[][] matrix2 = {
            {-5}
        };
        int k2 = 1;
        System.out.println(kthSmallest(matrix2, k2)); // Output: -5
    }
}



3.// Solution


import java.util.*;

public class KMostFrequentWords {
    public static List<String> kMostFrequent(String[] words, int k) {
        // Count the frequency of each word using a HashMap
        Map<String, Integer> wordFrequency = new HashMap<>();
        for (String word : words) {
            wordFrequency.put(word, wordFrequency.getOrDefault(word, 0) + 1);
        }

        // Custom comparator for sorting words by frequency and lexicographical order
        Comparator<String> customComparator = (a, b) -> {
            int freqA = wordFrequency.get(a);
            int freqB = wordFrequency.get(b);
            if (freqA != freqB) {
                return freqB - freqA; // Sort by frequency in descending order
            } else {
                return a.compareTo(b); // Sort by lexicographical order
            }
        };

        // Use a Priority Queue (Min Heap) to get the k most frequent words
        PriorityQueue<String> minHeap = new PriorityQueue<>(customComparator);

        for (String word : wordFrequency.keySet()) {
            minHeap.offer(word);
            if (minHeap.size() > k) {
                minHeap.poll(); // Remove the least frequent word
            }
        }

        // The result will be the k most frequent words in descending order
        List<String> result = new ArrayList<>();
        while (!minHeap.isEmpty()) {
            result.add(minHeap.poll());
        }
        Collections.reverse(result);

        return result;
    }

    public static void main(String[] args) {
        String[] words1 = {"i", "love", "leetcode", "i", "love", "coding"};
        int k1 = 2;
        System.out.println(kMostFrequent(words1, k1)); // Output: ["i", "love"]

        String[] words2 = {"the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"};
        int k2 = 4;
        System.out.println(kMostFrequent(words2, k2)); // Output: ["the", "is", "sunny", "day"]
    }
}



4.// Solution


class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int val) {
        this.val = val;
    }
}

public class BinaryTreeDiameter {
    public int diameterOfBinaryTree(TreeNode root) {
        if (root == null) {
            return 0;
        }

        // Calculate the diameter recursively
        int leftDiameter = diameterOfBinaryTree(root.left);
        int rightDiameter = diameterOfBinaryTree(root.right);

        // Calculate the height of the left and right subtrees
        int leftHeight = height(root.left);
        int rightHeight = height(root.right);

        // The diameter is the maximum of the three values
        return Math.max(Math.max(leftDiameter, rightDiameter), leftHeight + rightHeight);
    }

    private int height(TreeNode node) {
        if (node == null) {
            return 0;
        }
        return 1 + Math.max(height(node.left), height(node.right));
    }

    public static void main(String[] args) {
        // Example 1
        TreeNode root1 = new TreeNode(1);
        root1.left = new TreeNode(2);
        root1.right = new TreeNode(3);
        root1.left.left = new TreeNode(4);
        root1.left.right = new TreeNode(5);

        BinaryTreeDiameter binaryTreeDiameter = new BinaryTreeDiameter();
        System.out.println(binaryTreeDiameter.diameterOfBinaryTree(root1)); // Output: 3

        // Example 2
        TreeNode root2 = new TreeNode(1);
        root2.left = new TreeNode(2);

        System.out.println(binaryTreeDiameter.diameterOfBinaryTree(root2)); // Output: 1
    }
}



5.// Solution

Recursive Approach:-

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int val) {
        this.val = val;
    }
}

public class SymmetricBinaryTree {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isMirror(root.left, root.right);
    }

    private boolean isMirror(TreeNode node1, TreeNode node2) {
        if (node1 == null && node2 == null) {
            return true;
        }
        if (node1 == null || node2 == null || node1.val != node2.val) {
            return false;
        }
        return isMirror(node1.left, node2.right) && isMirror(node1.right, node2.left);
    }

    public static void main(String[] args) {
        // Example 1
        TreeNode root1 = new TreeNode(1);
        root1.left = new TreeNode(2);
        root1.right = new TreeNode(2);
        root1.left.left = new TreeNode(3);
        root1.left.right = new TreeNode(4);
        root1.right.left = new TreeNode(4);
        root1.right.right = new TreeNode(3);

        SymmetricBinaryTree symmetricBinaryTree = new SymmetricBinaryTree();
        System.out.println(symmetricBinaryTree.isSymmetric(root1)); // Output: true

        // Example 2
        TreeNode root2 = new TreeNode(1);
        root2.left = new TreeNode(2);
        root2.right = new TreeNode(2);
        root2.left.right = new TreeNode(3);
        root2.right.right = new TreeNode(3);

        System.out.println(symmetricBinaryTree.isSymmetric(root2)); // Output: false
    }
}



Iterative approach:-

import java.util.LinkedList;
import java.util.Queue;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int val) {
        this.val = val;
    }
}

public class SymmetricBinaryTree {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root.left);
        queue.offer(root.right);

        while (!queue.isEmpty()) {
            TreeNode node1 = queue.poll();
            TreeNode node2 = queue.poll();

            if (node1 == null && node2 == null) {
                continue;
            }
            if (node1 == null || node2 == null || node1.val != node2.val) {
                return false;
            }

            queue.offer(node1.left);
            queue.offer(node2.right);
            queue.offer(node1.right);
            queue.offer(node2.left);
        }

        return true;
    }

    public static void main(String[] args) {
        // Example 1
        TreeNode root1 = new TreeNode(1);
        root1.left = new TreeNode(2);
        root1.right = new TreeNode(2);
        root1.left.left = new TreeNode(3);
        root1.left.right = new TreeNode(4);
        root1.right.left = new TreeNode(4);
        root1.right.right = new TreeNode(3);

        SymmetricBinaryTree symmetricBinaryTree = new SymmetricBinaryTree();
        System.out.println(symmetricBinaryTree.isSymmetric(root1)); // Output: true

        // Example 2
        TreeNode root2 = new TreeNode(1);
        root2.left = new TreeNode(2);
        root2.right = new TreeNode(2);
        root2.left.right = new TreeNode(3);
        root2.right.right = new TreeNode(3);

        System.out.println(symmetricBinaryTree.isSymmetric(root2)); // Output: false
    }
}



6.// Solution


import java.util.PriorityQueue;

public class KthLargestInteger {
    public String kthLargestNumber(String[] nums, int k) {
        // Custom comparator to compare integers represented by strings
        PriorityQueue<String> minHeap = new PriorityQueue<>((a, b) -> {
            if (a.length() != b.length()) {
                return a.length() - b.length();
            }
            return a.compareTo(b);
        });

        for (String num : nums) {
            minHeap.offer(num);
            if (minHeap.size() > k) {
                minHeap.poll(); // Remove the smallest element if the size exceeds k
            }
        }

        return minHeap.peek();
    }

    public static void main(String[] args) {
        // Example 1
        String[] nums1 = {"3", "6", "7", "10"};
        int k1 = 4;
        KthLargestInteger kthLargestInteger = new KthLargestInteger();
        System.out.println(kthLargestInteger.kthLargestNumber(nums1, k1)); // Output: "3"

        // Example 2
        String[] nums2 = {"2", "21", "12", "1"};
        int k2 = 3;
        System.out.println(kthLargestInteger.kthLargestNumber(nums2, k2)); // Output: "2"

        // Example 3
        String[] nums3 = {"0", "0"};
        int k3 = 2;
        System.out.println(kthLargestInteger.kthLargestNumber(nums3, k3)); // Output: "0"
    }
}



7.// Solution


class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int val) {
        this.val = val;
    }
}

public class InvertBinaryTree {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }

        // Swap the left and right children of the current node
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        // Recursively invert the left and right subtrees
        invertTree(root.left);
        invertTree(root.right);

        return root;
    }

    public static void main(String[] args) {
        // Example 1
        TreeNode root1 = new TreeNode(4);
        root1.left = new TreeNode(2);
        root1.right = new TreeNode(7);
        root1.left.left = new TreeNode(1);
        root1.left.right = new TreeNode(3);
        root1.right.left = new TreeNode(6);
        root1.right.right = new TreeNode(9);

        InvertBinaryTree invertBinaryTree = new InvertBinaryTree();
        TreeNode invertedRoot1 = invertBinaryTree.invertTree(root1);
        printTree(invertedRoot1); // Output: [4, 7, 2, 9, 6, 3, 1]

        // Example 2
        TreeNode root2 = new TreeNode(2);
        root2.left = new TreeNode(1);
        root2.right = new TreeNode(3);

        TreeNode invertedRoot2 = invertBinaryTree.invertTree(root2);
        printTree(invertedRoot2); // Output: [2, 3, 1]

        // Example 3
        // Empty tree, no need to invert
        TreeNode root3 = null;
        TreeNode invertedRoot3 = invertBinaryTree.invertTree(root3);
        printTree(invertedRoot3); // Output: []
    }

    // Helper method to print the tree in a pre-order traversal (for validation)
    private static void printTree(TreeNode root) {
        if (root == null) {
            System.out.print("null ");
            return;
        }
        System.out.print(root.val + " ");
        printTree(root.left);
        printTree(root.right);
    }
}



8.// Solution


import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode(int val) {
        this.val = val;
    }
}

public class BinaryTreeLayoutMatrix {
    public List<List<String>> printTree(TreeNode root) {
        // Get the height of the tree
        int height = getHeight(root);

        // Calculate the dimensions of the matrix
        int rows = height + 1;
        int cols = (1 << height) - 1;

        // Initialize the matrix with empty strings
        List<List<String>> matrix = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            List<String> row = new ArrayList<>();
            for (int j = 0; j < cols; j++) {
                row.add("");
            }
            matrix.add(row);
        }

        // Perform level-order traversal to fill the matrix
        Queue<TreeNode> queue = new LinkedList<>();
        Queue<int[]> indexQueue = new LinkedList<>();
        queue.offer(root);
        indexQueue.offer(new int[]{0, (cols - 1) / 2});

        for (int row = 0; row < rows; row++) {
            int levelSize = queue.size();
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                int[] indices = indexQueue.poll();
                if (node != null) {
                    int col = indices[1];
                    matrix.get(row).set(col, String.valueOf(node.val));

                    if (node.left != null) {
                        queue.offer(node.left);
                        indexQueue.offer(new int[]{row + 1, col - (1 << (rows - row - 2))});
                    }
                    if (node.right != null) {
                        queue.offer(node.right);
                        indexQueue.offer(new int[]{row + 1, col + (1 << (rows - row - 2))});
                    }
                }
            }
        }

        return matrix;
    }

    private int getHeight(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(getHeight(root.left), getHeight(root.right));
    }

    public static void main(String[] args) {
        // Example 1
        TreeNode root1 = new TreeNode(1);
        root1.left = new TreeNode(2);

        BinaryTreeLayoutMatrix binaryTreeLayoutMatrix = new BinaryTreeLayoutMatrix();
        List<List<String>> matrix1 = binaryTreeLayoutMatrix.printTree(root1);
        for (List<String> row : matrix1) {
            System.out.println(row);
        }
        /*
         Output:
         ["", "1", ""]
         ["2", "", ""]
        */

        // Example 2
        TreeNode root2 = new TreeNode(1);
        root2.left = new TreeNode(2);
        root2.right = new TreeNode(3);
        root2.right.left = new TreeNode(4);

        List<List<String>> matrix2 = binaryTreeLayoutMatrix.printTree(root2);
        for (List<String> row : matrix2) {
            System.out.println(row);
        }
        /*
         Output:
         ["", "", "", "1", "", "", ""]
         ["", "2", "", "", "", "3", ""]
         ["", "", "4", "", "", "", ""]
        */
    }
}
