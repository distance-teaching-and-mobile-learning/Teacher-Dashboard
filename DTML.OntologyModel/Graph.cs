using ProtoBuf;
using System.Collections.Generic;

namespace DTML.Common.Models
{
    public class GraphInfo
    {
        public int word_count { get; set; }
        public int lemmas_count { get; set; }
        public int edges_count { get; set; }

        public Dictionary<string, int?> parts_of_speech { get; set; }
    }

    [ProtoContract]
    public class Graph
    {
        [ProtoMember(1, OverwriteList = true)]
        public List<Node> nodes { get; set; }

        [ProtoMember(2, OverwriteList = true)]
        public List<Edge> edges { get; set; }
    }

    [ProtoContract]
    public class Node
    {
        [ProtoMember(1)]
        public string id { get; set; }

        [ProtoMember(2)]
        public int cluster { get; set; }

        [ProtoMember(3)]
        public string word { get; set; }

        [ProtoMember(4)]
        public int size { get; set; }

        [ProtoMember(5)]
        public bool root { get; set; }

        [ProtoMember(6)]
        public string pos { get; set; }
    }

    [ProtoContract]
    public class Edge
    {
        [ProtoMember(1)]
        public string source { get; set; }

        [ProtoMember(2)]
        public string target { get; set; }

        [ProtoMember(3)]
        public string distance { get; set; }
    }
}
