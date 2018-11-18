using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace romi
{
    class Program
    {
        static void Main(string[] args)
        {
            int num;
            int x = 3000;
            Console.WriteLine("Enter number");
            num = int.Parse(Console.ReadLine());
            if (num == 70)
            {
                Console.WriteLine("70");

            }
            else if (num > 60)
            {
                Console.WriteLine("Pass");
                x = 1600;
            }
            else if (num < 60)
                x = 2600;
            else
                x = 0;
            Console.WriteLine("x={0}",x);

            Console.ReadLine();
            
        }
    }
}
