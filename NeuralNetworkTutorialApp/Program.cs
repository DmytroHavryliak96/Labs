using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork;

namespace NeuralNetworkTutorialApp
{
    class Program
    {
        static void Main(string[] args)
        {
            int[] layerSizes = new int[3] { 1, 2, 1 };
            TransferFunction[] TFuncs = new TransferFunction[3] {TransferFunction.None,
                                                               TransferFunction.Sigmoid,
                                                               TransferFunction.Sigmoid};

            BackPropagationNetwork bpn = new BackPropagationNetwork(layerSizes, TFuncs);

            double[] input = new double[] {0.0 };
            double[] desired = new double[1] {0.1};
            double[] output = new double[1];

            double error = 0.0;

            for(int i = 0; i < 1000; i++)
            {
                error = bpn.Train(ref input, ref desired, 0.15, 0.1);
                bpn.Run(ref input, out output);
                if (i % 100 == 0)
                    Console.WriteLine("Iteration {0}: \n\t Input {1:0.000} Output {2:0.000} Error{3:0.000}", i, input[0], output[0], error);
            }

            Console.ReadKey();
        }
    }
}
